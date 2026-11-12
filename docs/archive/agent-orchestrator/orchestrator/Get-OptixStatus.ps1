# Optix status monitor. Read-only: shows tasks, lanes, and daily budgets.
# Usage: .\Get-OptixStatus.ps1 [-SkipLaneCheck]

param([switch]$SkipLaneCheck)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib\State.ps1')
. (Join-Path $PSScriptRoot 'adapters\Muse.ps1')
. (Join-Path $PSScriptRoot 'adapters\Codex.ps1')
. (Join-Path $PSScriptRoot 'adapters\Copilot.ps1')

$config = ConvertFrom-Json (Get-Content (Join-Path $PSScriptRoot 'optix.config.json') -Raw)
$stateDir = Get-OptixStateDir $PSScriptRoot
$state = Read-OptixState $stateDir

Write-Output '== Optix tasks =='
if ($state.tasks.Count -eq 0) {
    Write-Output '(none yet; submit one with Submit-Task.ps1)'
} else {
    $rows = @()
    foreach ($k in ($state.tasks.Keys | Sort-Object { [int]$_ })) {
        $t = $state.tasks[$k]
        $rows += (New-Object psobject -Property @{
            Issue = $t.issue; Title = $t.title; Lane = $t.lane; Stage = $t.stage;
            Attempt = $t.attempt; PR = $t.pr; Updated = $t.updated
        })
    }
    $rows | Format-Table Issue, Title, Lane, Stage, Attempt, Updated -AutoSize | Out-String | Write-Output
    foreach ($k in ($state.tasks.Keys | Sort-Object { [int]$_ })) {
        $t = $state.tasks[$k]
        if ($t.pr) { Write-Output ("  #{0} PR: {1}" -f $t.issue, $t.pr) }
        if ($t.outcome) { Write-Output ("  #{0} outcome: {1}" -f $t.issue, $t.outcome) }
    }
}

if (-not $SkipLaneCheck) {
    Write-Output '== Lanes =='
    $lanes = @(
        (New-Object psobject -Property @{ Lane = 'muse'; Status = (Test-MuseLane $config.lanes.muse) }),
        (New-Object psobject -Property @{ Lane = 'codex'; Status = (Test-CodexLane $config.lanes.codex) }),
        (New-Object psobject -Property @{ Lane = 'copilot'; Status = (Test-CopilotLane $config.lanes.copilot) })
    )
    foreach ($l in $lanes) {
        $state2 = 'disabled'
        if ($l.Status.Enabled) { $state2 = 'ENABLED' }
        Write-Output ("  {0}: {1} ({2})" -f $l.Lane, $state2, $l.Status.Reason)
        if ($l.Status.Version) { Write-Output ("    {0}" -f $l.Status.Version) }
    }
}

Write-Output '== Daily budgets =='
Write-Output ("  date: {0}" -f $state.daily.date)
foreach ($name in @('muse', 'codex', 'copilot')) {
    $used = 0
    if ($state.daily.counts.ContainsKey($name)) { $used = [int]$state.daily.counts[$name] }
    Write-Output ("  {0}: {1}/{2}" -f $name, $used, $config.budgets.$name.tasksPerDay)
}
