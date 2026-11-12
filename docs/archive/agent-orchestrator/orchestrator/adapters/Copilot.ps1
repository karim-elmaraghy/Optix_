# Optix Copilot worker lane. Disabled until the CLI is installed and its
# non-interactive interface is verified. Never pretends to run.

function Test-CopilotLane {
    param($Lane)
    if (-not $Lane.enabled) {
        $reason = 'disabled in config'
        if ($Lane.reason) { $reason += (': {0}' -f $Lane.reason) }
        $found = Get-Command copilot -ErrorAction SilentlyContinue
        if ($found) {
            $probe = ''
            try { $probe = ((& copilot --help 2>&1 | Out-String)) } catch { $probe = $_.Exception.Message }
            if ($probe -match 'was not found') { $reason += ' (only VS Code shim present; real CLI not installed)' }
            else { $reason += ' (executable present but interface not verified)' }
        }
        return (New-Object psobject -Property @{ Enabled = $false; Reason = $reason; Exe = $null; Version = $null })
    }
    $cmd = Get-Command copilot -ErrorAction SilentlyContinue
    if (-not $cmd) {
        return (New-Object psobject -Property @{ Enabled = $false; Reason = 'copilot executable not found'; Exe = $null; Version = $null })
    }
    return (New-Object psobject -Property @{ Enabled = $false; Reason = 'copilot interface not verified (no verified non-interactive entrypoint yet)'; Exe = $cmd.Source; Version = $null })
}

function Invoke-CopilotWorker {
    throw 'copilot lane is disabled: CLI interface not verified'
}
