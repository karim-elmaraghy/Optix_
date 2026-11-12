# Optix single-pass dispatcher. Claims one ready task, builds an isolated
# worktree + branch, runs the worker, verifies the result independently, and
# opens a draft PR. Never merges, deploys, force-pushes, or deletes.
# Usage:
#   .\Dispatch.ps1 -Issue 12
#   .\Dispatch.ps1                    (oldest ready, unblocked task)
#   .\Dispatch.ps1 -Issue 12 -Retry   (retry a failed task within its budget)
#   .\Dispatch.ps1 -Issue 12 -WhatIf  (plan only, no writes)

param(
    [int]$Issue = 0,
    [string]$WorktreeRoot = '',
    [int]$TimeoutMin = 0,
    [switch]$Retry,
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib\Spec.ps1')
. (Join-Path $PSScriptRoot 'lib\Github.ps1')
. (Join-Path $PSScriptRoot 'lib\State.ps1')
. (Join-Path $PSScriptRoot 'lib\Process.ps1')
. (Join-Path $PSScriptRoot 'lib\Git.ps1')
. (Join-Path $PSScriptRoot 'adapters\Muse.ps1')
. (Join-Path $PSScriptRoot 'adapters\Codex.ps1')
. (Join-Path $PSScriptRoot 'adapters\Copilot.ps1')

function Get-LaneStatus {
    param($Config, [string]$Name)
    if ($Name -eq 'muse') { return (Test-MuseLane $Config.lanes.muse) }
    if ($Name -eq 'codex') { return (Test-CodexLane $Config.lanes.codex) }
    return (Test-CopilotLane $Config.lanes.copilot)
}

function Test-TaskBlocked {
    param($Repo, [int[]]$BlockedBy)
    foreach ($b in $BlockedBy) {
        $iss = Get-OptixIssue -Repo $Repo -Number $b
        if ($iss.state -ne 'closed') { return $b }
    }
    return 0
}

function Get-BudgetCount {
    param([hashtable]$State, [string]$Lane)
    $today = (Get-Date).ToString('yyyy-MM-dd')
    if ($State.daily.date -ne $today) {
        $State.daily.date = $today
        $State.daily.counts = @{}
    }
    if (-not $State.daily.counts.ContainsKey($Lane)) { return 0 }
    return [int]$State.daily.counts[$Lane]
}

function New-TaskSlug {
    param([string]$Title)
    $s = ($Title.ToLower() -replace '[^a-z0-9]+', '-').Trim('-')
    if ($s.Length -gt 40) { $s = $s.Substring(0, 40).Trim('-') }
    if ($s -eq '') { $s = 'task' }
    return $s
}

function Get-WorktreeChanges {
    param([string]$RepoRoot, [string]$Wt)
    $porc = Invoke-RepoGit $RepoRoot @('-C', $Wt, 'status', '--porcelain=v1') @($Wt)
    $files = @()
    foreach ($l in $porc) {
        if ($l.Length -lt 4) { continue }
        $p = $l.Substring(3).Trim()
        if ($p -match '^"(.*)"$') { $p = $Matches[1] }
        if ($p -match '^(.*) -> (.*)$') { $p = $Matches[2] }
        $files += ($p -replace '\\', '/')
    }
    return (@($files | Sort-Object -Unique))
}

$config = ConvertFrom-Json (Get-Content (Join-Path $PSScriptRoot 'optix.config.json') -Raw)
$repoRoot = Get-OptixRepoRoot $PSScriptRoot
$stateDir = Get-OptixStateDir $PSScriptRoot
$state = Read-OptixState $stateDir

# Pick the task.
if ($Issue -gt 0) {
    $key = ("{0}" -f $Issue)
    if (-not $state.tasks.ContainsKey($key)) { throw ("issue #{0} has no Optix state; submit it via Submit-Task first" -f $Issue) }
} else {
    $key = $null
    foreach ($k in ($state.tasks.Keys | Sort-Object { [int]$_ })) {
        if ($state.tasks[$k].stage -eq 'ready') { $key = $k; break }
    }
    if (-not $key) { Write-Output 'No ready tasks.'; exit 0 }
}
$task = $state.tasks[$key]
$num = [int]$key
$specText = Get-Content (Join-Path $stateDir $task.specFile) -Raw -ErrorAction Stop
$spec = Parse-OptixSpec $specText
if ($spec.Errors.Count -gt 0) { throw ('stored spec for #{0} no longer parses: {1}' -f $num, ($spec.Errors -join '; ')) }
$runDir = Get-OptixRunDir $stateDir $num

# Retry handling: rewind failed tasks within budget.
if ($Retry) {
    if ($task.stage -notlike 'failed-*') { throw ("issue #{0} is '{1}', nothing to retry" -f $num, $task.stage) }
    $cap = $spec.Retries
    if ($cap -lt 0) { $cap = $config.defaultRetries }
    if ($task.attempt -gt $cap) { throw ("issue #{0} exhausted retries ({1} attempts, cap {2})" -f $num, $task.attempt, ($cap + 1)) }
    $task.attempt++
    if ($task.stage -eq 'failed-push') { $task.stage = 'verified' } else { $task.stage = 'claimed' }
    $task.outcome = ''
    Write-OptixState $stateDir $state
    Write-Output ("Retrying issue #{0}, attempt {1}." -f $num, $task.attempt)
}

# Blocked-by check (fresh from GitHub).
$blocker = Test-TaskBlocked $config.repo $spec.BlockedBy
if ($blocker -ne 0) {
    Write-Output ("issue #{0} blocked by open #{1}; leaving for later." -f $num, $blocker)
    exit 0
}

# Pick the lane.
$lane = ''
if ($spec.Agent -ne 'any') {
    $st = Get-LaneStatus $config $spec.Agent
    if (-not $st.Enabled) { throw ("requested lane '{0}' unavailable: {1}" -f $spec.Agent, $st.Reason) }
    $lane = $spec.Agent
} else {
    foreach ($name in @($config.workerOrder)) {
        $st = Get-LaneStatus $config $name
        if (-not $st.Enabled) { Write-Output ("lane {0} skipped: {1}" -f $name, $st.Reason); continue }
        $cap = $config.budgets.$name.tasksPerDay
        if ((Get-BudgetCount $state $name) -ge $cap) { Write-Output ("lane {0} skipped: daily budget exhausted" -f $name); continue }
        $lane = $name
        break
    }
    if ($lane -eq '') { throw 'no enabled lane with remaining budget' }
}
$laneStatus = Get-LaneStatus $config $lane
$timeout = $TimeoutMin
if ($timeout -le 0) { $timeout = $spec.TimeoutMin }
if ($timeout -le 0) { $timeout = $config.defaultTimeoutMin }
$maxSteps = $spec.MaxSteps
if ($maxSteps -le 0) { $maxSteps = $config.defaultMaxModelSteps }
$slug = New-TaskSlug $spec.Title
$branch = ("{0}-{1}" -f $num, $slug)
$wtRoot = $WorktreeRoot
if ($wtRoot -eq '') { $wtRoot = $config.worktreeRoot }
if (-not [IO.Path]::IsPathRooted($wtRoot)) { $wtRoot = Join-Path $repoRoot $wtRoot }
$wt = [IO.Path]::GetFullPath((Join-Path $wtRoot $branch))

if ($WhatIf) {
    Write-Output ("WhatIf: issue #{0} on lane {1}" -f $num, $lane)
    Write-Output ("  branch:  {0}" -f $branch)
    Write-Output ("  worktree:{0}" -f $wt)
    Write-Output ("  timeout: {0} min, max steps: {1}" -f $timeout, $maxSteps)
    Write-Output ("  files:   {0}" -f ($spec.ExpectFiles -join ', '))
    exit 0
}

# Single-instance lock.
$lock = Test-OptixLock $stateDir
if ($lock -and $lock -ne 'stale') { throw ("dispatcher already running (pid {0}); refusing second instance" -f $lock.pid) }
if ($lock -eq 'stale') { Write-Output 'Removing stale dispatcher lock.' }
Set-OptixLock $stateDir
try {
    $markerBase = ("<!-- optix:{0}" -f $num)

    # Claim.
    if ($task.stage -eq 'ready') {
        $labels = Get-OptixIssueLabels -Repo $config.repo -Number $num
        if ($labels -notcontains 'status:ready') { throw ("issue #{0} no longer ready; refusing to claim" -f $num) }
        Set-OptixStatusLabel -Repo $config.repo -Number $num -Status 'status:claimed'
        Set-OptixAgentLabel -Repo $config.repo -Number $num -Lane $lane
        New-OptixCommentOnce -Repo $config.repo -Number $num -Marker ("{0}:claimed:1 -->" -f $markerBase) -Body ("Optix claimed this task on lane {0} (attempt 1)." -f $lane) | Out-Null
        $task.stage = 'claimed'
        $task.attempt = 1
        $task.lane = $lane
        $task.branch = $branch
        $task.worktree = $wt
        $today = (Get-Date).ToString('yyyy-MM-dd')
        if ($state.daily.date -ne $today) { $state.daily.date = $today; $state.daily.counts = @{} }
        if (-not $state.daily.counts.ContainsKey($lane)) { $state.daily.counts[$lane] = 0 }
        $state.daily.counts[$lane] = [int]$state.daily.counts[$lane] + 1
        $task.updated = (Get-Date).ToString('o')
        Write-OptixState $stateDir $state
    }
    $lane = $task.lane
    $branch = $task.branch
    $wt = $task.worktree

    # Worktree + branch (resume if already ours).
    if ($task.stage -eq 'claimed') {
        $haveBranch = $true
        try { Invoke-RepoGit $repoRoot @('show-ref', '--verify', '--quiet', ("refs/heads/{0}" -f $branch)) | Out-Null } catch { $haveBranch = $false }
        $wtList = Invoke-RepoGit $repoRoot @('worktree', 'list', '--porcelain')
        $wtNorm = ($wt -replace '\\', '/')
        $haveWt = (@($wtList -match [regex]::Escape($wtNorm)).Count -gt 0)
        if ($haveBranch -and -not $haveWt) { throw ("branch {0} exists without our worktree; refusing to adopt it" -f $branch) }
        if (-not $haveBranch) {
            Invoke-RepoGit $repoRoot @('worktree', 'add', '-b', $branch, $wt, $config.baseBranch) | Out-Null
            Write-Output ("Created worktree {0} on branch {1}." -f $wt, $branch)
        } else {
            Write-Output 'Resuming in existing worktree.'
        }
        try { $who = Invoke-RepoGit $repoRoot @('-C', $wt, 'config', 'user.name') @($wt) } catch { $who = '' }
        if (($who -join '').Trim() -eq '') {
            try {
                Invoke-RepoGit $repoRoot @('-C', $wt, 'config', 'user.name', 'Optix Worker') @($wt) | Out-Null
                Invoke-RepoGit $repoRoot @('-C', $wt, 'config', 'user.email', 'optix-worker@local') @($wt) | Out-Null
                $check = Invoke-RepoGit $repoRoot @('-C', $wt, 'config', 'user.name') @($wt)
                if (($check -join '').Trim() -eq '') { throw 'user.name still empty after set' }
            } catch {
                throw ("worktree commit identity setup failed in {0}: {1}" -f $wt, $_.Exception.Message)
            }
            Write-Output 'Set worktree-local commit identity (no global identity configured).'
        }
        $task.stage = 'worktree-ready'
        $task.updated = (Get-Date).ToString('o')
        Write-OptixState $stateDir $state
    }

    # Worker (Muse only in Phase 1; other lanes throw by design).
    if ($task.stage -eq 'worktree-ready' -or $task.stage -eq 'claimed') {
        if ($lane -ne 'muse') { throw ("lane {0} has no Phase 1 worker" -f $lane) }
        $fresh = Test-MuseLane $config.lanes.muse
        if (-not $fresh.Enabled) { throw ("muse lane lost: {0}" -f $fresh.Reason) }
        $prompt = ('You are an Optix worker implementing one task. Work only inside this workspace. ' +
            "Task title: {0}. Goal: {1}. Context: {2}. Acceptance criteria: {3}. " -f $spec.Title, $spec.Goal, $spec.Context, ($spec.AcceptanceRaw -join ' | ')) +
            ("Only create or modify these files: {0}. " -f ($spec.ExpectFiles -join ', ')) +
            'Do not commit, push, open pull requests, or merge. Do not modify any other files. When done, print a short summary of the files you changed.'
        $res = Invoke-MuseWorker -Exe $fresh.Exe -Workspace $wt -Prompt $prompt -MaxSteps $maxSteps -Effort $config.reasoningEffort -ApprovalMode $config.approvalMode -TimeoutMin $timeout -LogDir $runDir -Banned @($config.lanes.muse.bannedFlags) -UseWorktreeExisting $fresh.SupportsWorktreeExisting
        $task.outcome = ("worker:{0}/exit:{1}" -f $res.Status, $res.ExitCode)
        if ($res.Status -eq 'timeout') {
            $task.stage = 'failed-timeout'
            $task.updated = (Get-Date).ToString('o')
            Write-OptixState $stateDir $state
            Set-OptixStatusLabel -Repo $config.repo -Number $num -Status 'status:failed'
            New-OptixCommentOnce -Repo $config.repo -Number $num -Marker ("{0}:timeout:{1} -->" -f $markerBase, $task.attempt) -Body ("Worker timed out after {0} minutes and was terminated. Logs kept locally." -f $timeout) | Out-Null
            Write-Output 'Worker timed out; task marked failed.'
            exit 2
        }
        $wjson = $null
        if (Test-Path $res.StdoutFile) { $wjson = Read-WorkerJson (Get-Content $res.StdoutFile -Raw -ErrorAction SilentlyContinue) }
        if (Test-WorkerApprovalNeeded $wjson) {
            $task.stage = 'blocked-approval'
            $task.updated = (Get-Date).ToString('o')
            Write-OptixState $stateDir $state
            Set-OptixStatusLabel -Repo $config.repo -Number $num -Status 'status:blocked'
            New-OptixCommentOnce -Repo $config.repo -Number $num -Marker ("{0}:approval:{1} -->" -f $markerBase, $task.attempt) -Body 'Worker stopped for an approval decision. A human must review the worktree and decide; the dispatcher will not approve on its own.' | Out-Null
            Write-Output 'Worker needs approval; task blocked for human decision.'
            exit 2
        }
        if ($res.ExitCode -ne 0) {
            $task.stage = 'failed-worker'
            $task.updated = (Get-Date).ToString('o')
            Write-OptixState $stateDir $state
            Set-OptixStatusLabel -Repo $config.repo -Number $num -Status 'status:failed'
            $tail = ''
            if (Test-Path $res.StderrFile) { $tail = ((Get-Content $res.StderrFile -Raw) -split "`r?`n" | Select-Object -Last 10) -join "`n" }
            New-OptixCommentOnce -Repo $config.repo -Number $num -Marker ("{0}:workerfail:{1} -->" -f $markerBase, $task.attempt) -Body ("Worker exited with code {0}. Stderr tail:`n``````n{1}`n``````" -f $res.ExitCode, $tail) | Out-Null
            Write-Output ("Worker failed with exit {0}." -f $res.ExitCode)
            exit 2
        }
        $task.stage = 'worker-done'
        $task.updated = (Get-Date).ToString('o')
        Write-OptixState $stateDir $state
    }

    # Independent verification (exit 0 is NOT enough).
    if ($task.stage -eq 'worker-done') {
        $verified = @()
        $unverified = @()
        $changed = Get-WorktreeChanges $repoRoot $wt
        $expected = @($spec.ExpectFiles | Sort-Object -Unique)
        $missing = @($expected | Where-Object { $changed -notcontains $_ })
        $extra = @($changed | Where-Object { $expected -notcontains $_ })
        if ($missing.Count -eq 0 -and $extra.Count -eq 0) { $verified += ("changed files match Expect-Files ({0})" -f ($changed -join ', ')) }
        else {
            if ($missing.Count -gt 0) { $unverified += ("missing expected files: {0}" -f ($missing -join ', ')) }
            if ($extra.Count -gt 0) { $unverified += ("unexpected changed files: {0}" -f ($extra -join ', ')) }
        }
        foreach ($c in $spec.Checks) {
            $fp = Join-Path $wt $c.File
            $content = $null
            if (Test-Path $fp) { $content = Get-Content $fp -Raw -ErrorAction SilentlyContinue }
            if (($content -ne $null) -and $content.Contains($c.Text)) {
                $verified += ("{0} contains required text" -f $c.File)
            } else {
                $unverified += ("{0} missing required text" -f $c.File)
            }
        }
        foreach ($m in $spec.ManualReview) { $unverified += ("human review needed: {0}" -f $m) }
        $testResult = @{ ran = $false; exit = $null; timedOut = $false }
        if ($spec.TestRun) {
            $tr = Invoke-OptixShellCommand -Command $spec.TestRun -Cwd $wt -TimeoutSec 300 -OutFile (Join-Path $runDir 'test.stdout.txt') -ErrFile (Join-Path $runDir 'test.stderr.txt')
            $testResult.ran = $true
            $testResult.exit = $tr.ExitCode
            $testResult.timedOut = $tr.TimedOut
            if (-not $tr.TimedOut -and $tr.ExitCode -eq 0) { $verified += 'test command passed' }
            else { $unverified += ("test command failed (exit {0}, timed out: {1})" -f $tr.ExitCode, $tr.TimedOut) }
        } else {
            $unverified += 'no automated test command in spec'
        }
        @{ verified = $verified; unverified = $unverified; tests = $testResult; changed = $changed; workerOutcome = $task.outcome } | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $runDir 'result.json') -Encoding UTF8
        $hardFail = ($missing.Count -gt 0) -or ($extra.Count -gt 0) -or ($testResult.ran -and ($testResult.timedOut -or $testResult.exit -ne 0))
        foreach ($c in $spec.Checks) {
            $fp = Join-Path $wt $c.File
            $content = $null
            if (Test-Path $fp) { $content = Get-Content $fp -Raw -ErrorAction SilentlyContinue }
            if (($content -eq $null) -or (-not $content.Contains($c.Text))) { $hardFail = $true }
        }
        if ($hardFail) {
            $task.stage = 'failed-verification'
            $task.updated = (Get-Date).ToString('o')
            Write-OptixState $stateDir $state
            Set-OptixStatusLabel -Repo $config.repo -Number $num -Status 'status:failed'
            New-OptixCommentOnce -Repo $config.repo -Number $num -Marker ("{0}:verifyfail:{1} -->" -f $markerBase, $task.attempt) -Body ("Verification failed.`nVerified: {0}`nNot verified: {1}" -f ($verified -join '; '), ($unverified -join '; ')) | Out-Null
            Write-Output 'Verification failed; task marked failed.'
            exit 2
        }
        $task.stage = 'verified'
        $task.updated = (Get-Date).ToString('o')
        Write-OptixState $stateDir $state
        Write-Output ("Verified ({0} checks); {1} item(s) flagged for human review." -f $verified.Count, ($unverified.Count))
    }

    # Commit, push, draft PR (idempotent; never merge).
    if ($task.stage -eq 'verified') {
        Invoke-RepoGit $repoRoot (@('-C', $wt, 'add', '--') + @($spec.ExpectFiles)) @($wt) | Out-Null
        $staged = $true
        try { Invoke-RepoGit $repoRoot @('-C', $wt, 'diff', '--cached', '--quiet') @($wt) | Out-Null; $staged = $false } catch { $staged = $true }
        if (-not $staged) { throw ("worker produced no staged changes in {0}" -f $wt) }
        Invoke-RepoGit $repoRoot @('-C', $wt, 'commit', '-m', ("optix: task #{0} {1}" -f $num, $slug)) @($wt) | Out-Null
        try {
            Invoke-RepoGit $repoRoot @('-C', $wt, 'push', '-u', 'origin', $branch) @($wt) | Out-Null
        } catch {
            $task.stage = 'failed-push'
            $task.updated = (Get-Date).ToString('o')
            Write-OptixState $stateDir $state
            Set-OptixStatusLabel -Repo $config.repo -Number $num -Status 'status:failed'
            New-OptixCommentOnce -Repo $config.repo -Number $num -Marker ("{0}:pushfail:{1} -->" -f $markerBase, $task.attempt) -Body 'Push failed (likely credentials). Commits are safe in the local worktree; fix access and rerun with -Retry.' | Out-Null
            Write-Output 'Push failed; task marked failed (retryable).'
            exit 2
        }
        $prNum = Get-OptixOpenPrForBranch -Repo $config.repo -Branch $branch
        $res2 = ConvertFrom-Json (Get-Content (Join-Path $runDir 'result.json') -Raw)
        $prBody = ("Optix task #{0} ({1}).`n`nVerified: {2}`nNeeds human review: {3}`n`nCloses #{0}" -f $num, $task.lane, (($res2.verified) -join '; '), (($res2.unverified) -join '; '))
        if ($prNum -ne '') {
            $prUrl = ("https://github.com/{0}/pull/{1}" -f $config.repo, $prNum)
            Write-Output ("Reusing existing PR #{0} for branch {1}." -f $prNum, $branch)
        } else {
            $pr = New-OptixDraftPr -Repo $config.repo -Title ("[optix #{0}] {1}" -f $num, $spec.Title) -Head $branch -Base $config.baseBranch -Body $prBody
            $prUrl = $pr.html_url
            Write-Output ("Opened draft PR: {0}" -f $prUrl)
        }
        Set-OptixStatusLabel -Repo $config.repo -Number $num -Status 'status:needs-review'
        New-OptixCommentOnce -Repo $config.repo -Number $num -Marker ("{0}:pr:{1} -->" -f $markerBase, $task.attempt) -Body ("Draft PR ready for human review and merge: {0}`nVerified: {1}`nNeeds human review: {2}" -f $prUrl, (($res2.verified) -join '; '), (($res2.unverified) -join '; ')) | Out-Null
        $task.stage = 'pr-open'
        $task.pr = $prUrl
        $task.updated = (Get-Date).ToString('o')
        Write-OptixState $stateDir $state
        Write-Output ("Done: {0} (not merged; awaiting your approval)." -f $prUrl)
    } else {
        Write-Output ("Nothing to do: stage is '{0}'." -f $task.stage)
    }
} finally {
    Remove-OptixLock $stateDir
}
