# Optix GitHub access. REST-only via `gh api` (GraphQL verified to 401 from some
# environments; always call from PowerShell, never Git bash). Ambient `gh` auth
# is used; no secrets are read or printed here. Request bodies travel as JSON
# via `gh api --input <file>` (UTF-8 temp file) so arbitrary user text never
# touches argv and needs no command-line escaping.

function Convert-OptixTypedValue {
    param([Parameter(Mandatory = $true)][string]$Text)
    if ($Text -ceq 'true') { return $true }
    if ($Text -ceq 'false') { return $false }
    if ($Text -ceq 'null') { return $null }
    $l = 0
    if ([long]::TryParse($Text, [ref]$l)) { return $l }
    $d = 0.0
    if ([double]::TryParse($Text, [ref]$d)) { return $d }
    return $Text
}

function Build-OptixJsonBody {
    param([hashtable]$Fields = @{}, [hashtable]$TypedFields = @{}, [hashtable]$Lists = @{})
    $body = @{}
    foreach ($k in $Fields.Keys) { $body[$k] = [string]$Fields[$k] }
    foreach ($k in $TypedFields.Keys) { $body[$k] = Convert-OptixTypedValue ([string]$TypedFields[$k]) }
    foreach ($k in $Lists.Keys) { $body[$k] = @($Lists[$k] | ForEach-Object { [string]$_ }) }
    return ($body | ConvertTo-Json -Compress -Depth 8)
}

function Get-OptixTmpDir {
    $d = Join-Path (Join-Path $PSScriptRoot '..\..') '.optix\tmp'
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
    return $d
}

function Write-OptixJsonInput {
    param([Parameter(Mandatory = $true)][string]$Dir, [Parameter(Mandatory = $true)][string]$JsonBody)
    if (-not (Test-Path $Dir)) { New-Item -ItemType Directory -Path $Dir -Force | Out-Null }
    $p = Join-Path $Dir ("ghinput-{0}-{1}.json" -f $PID, [DateTime]::UtcNow.Ticks)
    [IO.File]::WriteAllText($p, $JsonBody, (New-Object System.Text.UTF8Encoding $false))
    return $p
}

function Get-OptixGhArgv {
    param([string]$Endpoint, [string]$Method = 'GET', [string]$Jq = '', [string]$InputFile = '')
    $a = @('api', $Endpoint)
    if ($Method -ne 'GET') { $a += @('-X', $Method) }
    if ($Jq -ne '') { $a += @('--jq', $Jq) }
    if ($InputFile -ne '') { $a += @('--input', $InputFile) }
    return $a
}

function Invoke-OptixGhApi {
    param(
        [Parameter(Mandatory = $true)][string]$Endpoint,
        [string]$Method = 'GET',
        [hashtable]$Fields = @{},
        [hashtable]$TypedFields = @{},
        [hashtable]$Lists = @{},
        [string]$Jq = ''
    )
    $hasBody = ($Fields.Count -gt 0) -or ($TypedFields.Count -gt 0) -or ($Lists.Count -gt 0)
    if (-not $hasBody) {
        $out = & gh @(Get-OptixGhArgv -Endpoint $Endpoint -Method $Method -Jq $Jq) 2>&1
        if ($LASTEXITCODE -ne 0) { throw ('gh api {0} failed: {1}' -f $Endpoint, ($out -join "`n")) }
        return $out
    }
    $json = Build-OptixJsonBody -Fields $Fields -TypedFields $TypedFields -Lists $Lists
    $inFile = Write-OptixJsonInput -Dir (Get-OptixTmpDir) -JsonBody $json
    $out = & gh @(Get-OptixGhArgv -Endpoint $Endpoint -Method $Method -Jq $Jq -InputFile $inFile) 2>&1
    if ($LASTEXITCODE -ne 0) { throw ('gh api {0} failed (input kept at {1}): {2}' -f $Endpoint, $inFile, ($out -join "`n")) }
    Remove-Item $inFile -Force -ErrorAction SilentlyContinue
    return $out
}

function Get-OptixIssue {
    param([string]$Repo, [int]$Number)
    $json = Invoke-OptixGhApi -Endpoint ("repos/{0}/issues/{1}" -f $Repo, $Number)
    return (ConvertFrom-Json ($json -join "`n"))
}

function Get-OptixIssueLabels {
    param([string]$Repo, [int]$Number)
    return @(Invoke-OptixGhApi -Endpoint ("repos/{0}/issues/{1}/labels" -f $Repo, $Number) -Jq '.[].name')
}

function Add-OptixIssueLabels {
    param([string]$Repo, [int]$Number, [string[]]$Labels)
    Invoke-OptixGhApi -Endpoint ("repos/{0}/issues/{1}/labels" -f $Repo, $Number) -Method POST -Lists @{ labels = $Labels } | Out-Null
}

function Remove-OptixIssueLabel {
    param([string]$Repo, [int]$Number, [string]$Label)
    $enc = [uri]::EscapeDataString($Label)
    Invoke-OptixGhApi -Endpoint ("repos/{0}/issues/{1}/labels/{2}" -f $Repo, $Number, $enc) -Method DELETE | Out-Null
}

function Set-OptixStatusLabel {
    param([string]$Repo, [int]$Number, [string]$Status)
    $labels = Get-OptixIssueLabels -Repo $Repo -Number $Number
    foreach ($l in $labels) {
        if ($l -like 'status:*' -and $l -ne $Status) {
            Remove-OptixIssueLabel -Repo $Repo -Number $Number -Label $l
        }
    }
    if ($labels -notcontains $Status) {
        Add-OptixIssueLabels -Repo $Repo -Number $Number -Labels @($Status)
    }
}

function Set-OptixAgentLabel {
    param([string]$Repo, [int]$Number, [string]$Lane)
    $labels = Get-OptixIssueLabels -Repo $Repo -Number $Number
    foreach ($l in $labels) {
        if ($l -like 'agent:*' -and $l -ne ('agent:' + $Lane)) {
            Remove-OptixIssueLabel -Repo $Repo -Number $Number -Label $l
        }
    }
    if ($labels -notcontains ('agent:' + $Lane)) {
        Add-OptixIssueLabels -Repo $Repo -Number $Number -Labels @(('agent:' + $Lane))
    }
}

function New-OptixTaskIssue {
    param([string]$Repo, [string]$Title, [string]$Body, [string[]]$Labels)
    $json = Invoke-OptixGhApi -Endpoint ("repos/{0}/issues" -f $Repo) -Method POST -Fields @{ title = $Title; body = $Body }
    $issue = ConvertFrom-Json ($json -join "`n")
    if ($Labels.Count -gt 0) {
        Add-OptixIssueLabels -Repo $Repo -Number $issue.number -Labels $Labels
    }
    return $issue
}

function Get-OptixCommentBodies {
    param([string]$Repo, [int]$Number)
    return @(Invoke-OptixGhApi -Endpoint ("repos/{0}/issues/{1}/comments" -f $Repo, $Number) -Jq '.[].body')
}

# Posts once: skips when the hidden marker is already present (restart-safe).
function New-OptixCommentOnce {
    param([string]$Repo, [int]$Number, [string]$Marker, [string]$Body)
    foreach ($b in (Get-OptixCommentBodies -Repo $Repo -Number $Number)) {
        if ($b -like ('*{0}*' -f $Marker)) { return 'skipped-duplicate' }
    }
    Invoke-OptixGhApi -Endpoint ("repos/{0}/issues/{1}/comments" -f $Repo, $Number) -Method POST -Fields @{ body = ($Body + "`n`n" + $Marker) } | Out-Null
    return 'posted'
}

function Get-OptixOpenPrForBranch {
    param([string]$Repo, [string]$Branch)
    $owner = ($Repo -split '/')[0]
    $out = Invoke-OptixGhApi -Endpoint ("repos/{0}/pulls?state=open&head={1}:{2}" -f $Repo, $owner, $Branch) -Jq '.[0].number'
    return ($out -join '').Trim()
}

function New-OptixDraftPr {
    param([string]$Repo, [string]$Title, [string]$Head, [string]$Base, [string]$Body)
    $json = Invoke-OptixGhApi -Endpoint ("repos/{0}/pulls" -f $Repo) -Method POST -Fields @{ title = $Title; head = $Head; base = $Base; body = $Body } -TypedFields @{ draft = 'true' }
    return (ConvertFrom-Json ($json -join "`n"))
}
