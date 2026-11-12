# Optix Muse worker lane. Non-interactive via `muse exec`. Sandbox and
# approval protections are always preserved; banned flags are rejected.

function Resolve-MuseExe {
    param($Lane)
    $configured = $Lane.executable
    if ($configured) {
        $exists = $false
        try { $exists = Test-Path $configured -ErrorAction Stop } catch { $exists = $false }
        if ($exists) { return (Get-Item $configured).FullName }
    }
    $cmd = Get-Command muse -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cmd = Get-Command muse.cmd -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Test-MuseLane {
    param($Lane)
    if (-not $Lane.enabled) {
        return (New-Object psobject -Property @{ Enabled = $false; Reason = ('disabled in config: {0}' -f $Lane.reason); Exe = $null; Version = $null; SupportsWorktreeExisting = $false })
    }
    $exe = Resolve-MuseExe $Lane
    if (-not $exe) {
        return (New-Object psobject -Property @{ Enabled = $false; Reason = 'muse executable not found (config path missing/inaccessible and not on PATH)'; Exe = $Lane.executable; Version = $null; SupportsWorktreeExisting = $false })
    }
    $ver = ''
    try { $ver = ((& $exe --version 2>&1 | Out-String).Trim()) } catch {
        return (New-Object psobject -Property @{ Enabled = $false; Reason = ('muse --version failed: {0}' -f $_.Exception.Message); Exe = $exe; Version = $null; SupportsWorktreeExisting = $false })
    }
    $help = ''
    try {
        $help = (& $exe exec --help 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0) { throw ('exit {0}' -f $LASTEXITCODE) }
    } catch {
        return (New-Object psobject -Property @{ Enabled = $false; Reason = ('muse exec --help failed: {0}' -f $_.Exception.Message); Exe = $exe; Version = $ver; SupportsWorktreeExisting = $false })
    }
    $missing = @($Lane.requiredFlags | Where-Object { $help -notlike ('*{0}*' -f $_) })
    if ($missing.Count -gt 0) {
        return (New-Object psobject -Property @{ Enabled = $false; Reason = ('muse help lacks required flags: {0}' -f ($missing -join ', ')); Exe = $exe; Version = $ver; SupportsWorktreeExisting = $false })
    }
    $supportsWt = $false
    foreach ($f in @($Lane.optionalFlags)) {
        if ($f -eq '--worktree-existing' -and $help -like '*--worktree-existing*') { $supportsWt = $true }
    }
    $firstLine = (($ver -split "`r?`n")[0])
    return (New-Object psobject -Property @{ Enabled = $true; Reason = 'ok'; Exe = $exe; Version = $firstLine; SupportsWorktreeExisting = $supportsWt })
}

function Invoke-MuseWorker {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string]$Workspace,
        [Parameter(Mandatory = $true)][string]$Prompt,
        [Parameter(Mandatory = $true)][int]$MaxSteps,
        [Parameter(Mandatory = $true)][string]$Effort,
        [Parameter(Mandatory = $true)][string]$ApprovalMode,
        [Parameter(Mandatory = $true)][int]$TimeoutMin,
        [Parameter(Mandatory = $true)][string]$LogDir,
        [string[]]$Banned = @(),
        [bool]$UseWorktreeExisting = $false
    )
    $clean = $Prompt -replace '"', "'"
    $wArgs = @('exec', '--workspace', $Workspace)
    if ($UseWorktreeExisting) { $wArgs += '--worktree-existing' }
    $wArgs += @('--json', '--reasoning-effort', $Effort, '--max-model-steps', ("{0}" -f $MaxSteps), '--approval-mode', $ApprovalMode, $clean)
    foreach ($b in $Banned) {
        if ($wArgs -contains $b) { throw ("banned worker flag present: {0}" -f $b) }
    }
    if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
    $stdout = Join-Path $LogDir 'worker.stdout.txt'
    $stderr = Join-Path $LogDir 'worker.stderr.txt'
    $p = Start-OptixWorker -Exe $Exe -WorkerArgs $wArgs -Cwd $Workspace -StdoutFile $stdout -StderrFile $stderr
    if (-not $p.WaitForExit([int]([double]$TimeoutMin * 60000))) {
        Stop-OptixProcessTree -TargetPid $p.Id -Started $p.StartTime | Out-Null
        return @{ Status = 'timeout'; ExitCode = -1; StdoutFile = $stdout; StderrFile = $stderr }
    }
    return @{ Status = 'exited'; ExitCode = $p.ExitCode; StdoutFile = $stdout; StderrFile = $stderr }
}

function Read-WorkerJson {
    param([string]$Text)
    try { return (ConvertFrom-Json $Text -ErrorAction Stop) } catch { }
    $lines = @($Text -split "`r?`n" | Where-Object { $_.Trim() -ne '' })
    $from = [Math]::Max(0, $lines.Count - 3)
    for ($i = $lines.Count - 1; $i -ge $from; $i--) {
        try { return (ConvertFrom-Json $lines[$i] -ErrorAction Stop) } catch { }
    }
    return $null
}

function Test-WorkerApprovalNeeded {
    param($Json)
    if (-not $Json) { return $false }
    $s = ($Json | ConvertTo-Json -Depth 4 -Compress)
    return ($s -match '(?i)approval.{0,30}(requir|needed|pending|request)|needs.?approval')
}
