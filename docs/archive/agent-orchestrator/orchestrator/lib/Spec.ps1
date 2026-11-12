# Optix task-spec parsing and validation. Pure local logic, no network.
# Spec format (pasted ChatGPT output):
#   # Task: <title>                       (required)
#   Agent: muse|codex|copilot|any         (optional, default any)
#   Blocked-By: #12, #15                  (optional)
#   Timeout-Min: 30                       (optional)
#   Retries: 1                            (optional)
#   Max-Steps: 50                         (optional)
#   Fallback: yes|no                      (optional, default no)
#   Expect-Files: docs/a.md, docs/b.md    (required in Phase 1)
#   ## Goal                               (required)
#   ## Context                            (optional)
#   ## Acceptance criteria                (required; machine-checkable lines:
#                                          - [file:docs/a.md] contains "text")
#   ## Test plan                           (optional; automated line: run: <cmd>)

function Set-OptixSection {
    param([hashtable]$Spec, [string]$Name, [string[]]$Lines)
    $text = (($Lines -join "`n").Trim())
    switch ($Name) {
        'goal' { $Spec.Goal = $text }
        'context' { $Spec.Context = $text }
        'acceptance criteria' {
            foreach ($l in $Lines) {
                $t = $l.Trim()
                if ($t -eq '') { continue }
                $Spec.AcceptanceRaw += $t
                if ($t -match '^-\s*\[file:(.+?)\]\s+contains\s+"(.*?)"\s*$') {
                    $Spec.Checks += @{ File = ($Matches[1] -replace '\\', '/'); Text = $Matches[2] }
                } elseif ($t -match '^-') {
                    $Spec.ManualReview += $t
                }
            }
        }
        'test plan' {
            foreach ($l in $Lines) {
                if ($l -match '^\s*run:\s*(.+?)\s*$' -and -not $Spec.TestRun) {
                    $Spec.TestRun = $Matches[1]
                }
            }
        }
    }
}

function Parse-OptixSpec {
    param([Parameter(Mandatory = $true)][string]$Text)
    $spec = @{
        Title = $null; Agent = 'any'; BlockedBy = @(); TimeoutMin = 0; Retries = -1;
        MaxSteps = 0; Fallback = $false; ExpectFiles = @(); Goal = ''; Context = '';
        AcceptanceRaw = @(); Checks = @(); ManualReview = @(); TestRun = $null; Errors = @()
    }
    $lines = $Text -split "`r?`n"
    $section = $null
    $buf = @()
    $trailers = @{}
    foreach ($line in $lines) {
        if ($line -match '^\s*#\s*Task:\s*(.+?)\s*$') { $spec.Title = $Matches[1]; continue }
        if ($line -match '^\s*##\s*(.+?)\s*$') {
            if ($section) { Set-OptixSection $spec $section $buf }
            $section = $Matches[1].Trim().ToLower()
            $buf = @()
            continue
        }
        if ($section) { $buf += $line; continue }
        if ($line -match '^\s*([A-Za-z-]+)\s*:\s*(.*?)\s*$') { $trailers[$Matches[1].ToLower()] = $Matches[2] }
    }
    if ($section) { Set-OptixSection $spec $section $buf }

    if ($trailers.ContainsKey('agent')) {
        $a = $trailers['agent'].ToLower()
        if (@('muse', 'codex', 'copilot', 'any') -contains $a) { $spec.Agent = $a }
        else { $spec.Errors += ("Agent must be muse, codex, copilot, or any (got '{0}')" -f $trailers['agent']) }
    }
    if ($trailers.ContainsKey('blocked-by') -and $trailers['blocked-by'].Trim() -ne '') {
        foreach ($part in ($trailers['blocked-by'] -split '[,\s]+')) {
            if ($part -match '^#?(\d+)$') { $spec.BlockedBy += [int]$Matches[1] }
            elseif ($part.Trim() -ne '') { $spec.Errors += ("Blocked-By entry not an issue number: '{0}'" -f $part) }
        }
    }
    foreach ($key in @('timeout-min', 'retries', 'max-steps')) {
        if ($trailers.ContainsKey($key) -and $trailers[$key].Trim() -ne '') {
            $n = 0
            if ([int]::TryParse($trailers[$key], [ref]$n) -and $n -ge 0) {
                if ($key -eq 'timeout-min') { $spec.TimeoutMin = $n }
                if ($key -eq 'retries') { $spec.Retries = $n }
                if ($key -eq 'max-steps') { $spec.MaxSteps = $n }
            } else {
                $spec.Errors += ("{0} must be a non-negative integer (got '{1}')" -f $key, $trailers[$key])
            }
        }
    }
    if ($trailers.ContainsKey('fallback')) {
        $f = $trailers['fallback'].ToLower()
        if ($f -eq 'yes' -or $f -eq 'true') { $spec.Fallback = $true }
        elseif ($f -eq 'no' -or $f -eq 'false') { $spec.Fallback = $false }
        else { $spec.Errors += ("Fallback must be yes or no (got '{0}')" -f $trailers['fallback']) }
    }
    if ($trailers.ContainsKey('expect-files')) {
        foreach ($part in ($trailers['expect-files'] -split ',')) {
            $p = ($part.Trim() -replace '\\', '/').Trim('/')
            if ($p -eq '') { continue }
            if ($p -match '(^|/)\.\.(/|$)') { $spec.Errors += ("Expect-Files entry escapes the worktree: '{0}'" -f $part) }
            elseif ($p -match '^[A-Za-z]:') { $spec.Errors += ("Expect-Files entry must be relative: '{0}'" -f $part) }
            else { $spec.ExpectFiles += $p }
        }
    }

    if (-not $spec.Title) { $spec.Errors += 'Missing required title line: # Task: <title>' }
    if ($spec.Goal -eq '') { $spec.Errors += 'Missing required section: ## Goal' }
    if ($spec.AcceptanceRaw.Count -eq 0) { $spec.Errors += 'Missing required section: ## Acceptance criteria' }
    if ($spec.ExpectFiles.Count -eq 0) { $spec.Errors += 'Missing required trailer: Expect-Files: <relative paths>' }
    return $spec
}
