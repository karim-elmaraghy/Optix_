# Optix Codex worker lane. Disabled until the CLI is installed and its
# non-interactive interface is verified. Never pretends to run.

function Test-CodexLane {
    param($Lane)
    if (-not $Lane.enabled) {
        $reason = 'disabled in config'
        if ($Lane.reason) { $reason += (': {0}' -f $Lane.reason) }
        $found = Get-Command codex -ErrorAction SilentlyContinue
        if ($found) { $reason += ' (executable present but interface not verified)' }
        return (New-Object psobject -Property @{ Enabled = $false; Reason = $reason; Exe = $null; Version = $null })
    }
    $cmd = Get-Command codex -ErrorAction SilentlyContinue
    if (-not $cmd) {
        return (New-Object psobject -Property @{ Enabled = $false; Reason = 'codex executable not found'; Exe = $null; Version = $null })
    }
    return (New-Object psobject -Property @{ Enabled = $false; Reason = 'codex interface not verified (no verified non-interactive entrypoint yet)'; Exe = $cmd.Source; Version = $null })
}

function Invoke-CodexWorker {
    throw 'codex lane is disabled: CLI interface not verified'
}
