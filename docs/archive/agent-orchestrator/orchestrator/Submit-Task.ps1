# Optix single task-submission interface. Paste a ChatGPT-generated spec (text,
# file, or stdin); this validates it, creates the GitHub task issue, and
# records local state. No manual assignment or issue creation needed.
# Usage:
#   .\Submit-Task.ps1 -File spec.md
#   .\Submit-Task.ps1 -Spec $text
#   Get-Content spec.md -Raw | .\Submit-Task.ps1
#   .\Submit-Task.ps1 -File spec.md -WhatIf   (validate only, no writes)

param(
    [string]$Spec,
    [string]$File,
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib\Spec.ps1')
. (Join-Path $PSScriptRoot 'lib\Github.ps1')
. (Join-Path $PSScriptRoot 'lib\State.ps1')

$config = ConvertFrom-Json (Get-Content (Join-Path $PSScriptRoot 'optix.config.json') -Raw)
$stateDir = Get-OptixStateDir $PSScriptRoot

if ($File) { $text = Get-Content $File -Raw -ErrorAction Stop }
elseif ($Spec) { $text = $Spec }
elseif ([Console]::IsInputRedirected) { $text = [Console]::In.ReadToEnd() }
else { throw 'Provide -Spec text, -File path, or pipe the spec on stdin.' }

$parsed = Parse-OptixSpec $text
if ($parsed.Errors.Count -gt 0) {
    Write-Output 'Spec invalid, nothing created:'
    foreach ($e in $parsed.Errors) { Write-Output ("  - {0}" -f $e) }
    exit 1
}

Write-Output ("Title:    {0}" -f $parsed.Title)
Write-Output ("Agent:    {0}" -f $parsed.Agent)
Write-Output ("Files:    {0}" -f ($parsed.ExpectFiles -join ', '))
Write-Output ("Checks:   {0} machine, {1} manual-review" -f $parsed.Checks.Count, $parsed.ManualReview.Count)
if ($parsed.TestRun) { Write-Output ("Test run: {0}" -f $parsed.TestRun) }
if ($WhatIf) { Write-Output 'WhatIf: issue creation skipped.'; exit 0 }

$labels = @('task', 'status:ready', ('agent:' + $parsed.Agent))
$issue = New-OptixTaskIssue -Repo $config.repo -Title $parsed.Title -Body ($text.Trim() + "`n`nSubmitted via Optix Submit-Task.") -Labels $labels

$runDir = Get-OptixRunDir $stateDir $issue.number
Set-Content (Join-Path $runDir 'spec.md') $text.Trim() -Encoding UTF8
$state = Read-OptixState $stateDir
$now = (Get-Date).ToString('o')
$state.tasks[("{0}" -f $issue.number)] = (New-Object psobject -Property @{
    issue = $issue.number; title = $parsed.Title; lane = ''; requestedAgent = $parsed.Agent;
    stage = 'ready'; branch = ''; worktree = ''; attempt = 0; outcome = ''; pr = '';
    updated = $now; specFile = ("runs/{0}/spec.md" -f $issue.number)
})
Write-OptixState $stateDir $state

Write-Output ("Created task issue #{0}: {1}" -f $issue.number, $issue.html_url)
