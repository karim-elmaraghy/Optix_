# Integration regression test for gh --input file transport. No network, no
# credentials, no live GitHub calls.
# Proves: (1) the input file hits disk byte-exactly (UTF-8, no BOM); (2) a real
# child process reads it back byte-identically, and the JSON round-trips with
# exact text plus correct types; (3) the real argv builder emits only fixed
# control tokens, so user text can never become flags.
# Run: .\management\orchestrator\tests\Test-GhInput.ps1
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\lib\Github.ps1')

foreach ($fn in @('Build-OptixJsonBody', 'Write-OptixJsonInput', 'Get-OptixGhArgv')) {
    if (-not (Get-Command $fn -ErrorAction SilentlyContinue)) {
        Write-Output ("FAIL: {0} not found; --input fix not applied." -f $fn)
        exit 1
    }
}

$stub = Join-Path $PSScriptRoot 'Read-InputFile.ps1'
$tmp = Join-Path $PSScriptRoot ("..\..\.optix\tmp-ghinput-{0}" -f $PID)
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory -Path $tmp | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding $false
$fail = 0
try {
    $child = (Get-Command powershell.exe -ErrorAction Stop).Source
    $uni = 'caf' + [char]0xE9 + ' na' + [char]0xEF + 've ' + [char]0x65E5 + [char]0x672C + [char]0xD83D + [char]0xDE00
    $cases = @(
        @{ Name = 'double-quotes'; Text = 'He said "ship --gate1 now" twice' },
        @{ Name = 'single-quotes'; Text = "Karim's review: it's fine" },
        @{ Name = 'markdown'; Text = "# Title`n- [file:x] contains `"code`"`n[link](https://x/y?a=1&b=2)" },
        @{ Name = 'multiline'; Text = "# Task: Demo`n`n## Goal`nDo it.`n`nDone.`n" },
        @{ Name = 'backslashes'; Text = 'C:\path\to\file plus trailing\' },
        @{ Name = 'json-fragment'; Text = '{"dtype":"float16","batch":4}' },
        @{ Name = 'unicode'; Text = $uni },
        @{ Name = 'trailing-whitespace'; Text = ('ends with spaces and tab   ' + "`t") },
        @{ Name = 'combined'; Text = "# Task: Demo (Karim's)`n`n- [file:run.py] contains `"--gate1`"`n- path C:\tmp\x plus 100% & done" }
    )
    foreach ($case in $cases) {
        $json = Build-OptixJsonBody -Fields @{ title = 't'; body = $case.Text } -TypedFields @{ draft = 'true' } -Lists @{ labels = @('task', 'status:ready') }
        $inFile = Write-OptixJsonInput -Dir $tmp -JsonBody $json
        $ok = $true
        $why = ''
        $raw = [IO.File]::ReadAllBytes($inFile)
        if ($raw.Length -eq 0 -or $raw[0] -ne 0x7B) { $ok = $false; $why = 'leading byte (BOM?)' }
        if ($utf8.GetString($raw) -cne $json) { $ok = $false; $why = 'file bytes exact' }
        $copyFile = $inFile + '.child.bin'
        & $child -NoProfile -NonInteractive -File $stub $inFile $copyFile
        if ($LASTEXITCODE -ne 0) { $ok = $false; $why = 'child exit' }
        else {
            $copy = [IO.File]::ReadAllBytes($copyFile)
            if (($copy.Length -ne $raw.Length) -or (($copy -join ',') -cne ($raw -join ','))) {
                $ok = $false; $why = 'child bytes identical'
            }
        }
        $parsed = ConvertFrom-Json $json
        if ($parsed.body -cne $case.Text) { $ok = $false; $why = 'body round-trip' }
        if ($parsed.title -cne 't') { $ok = $false; $why = 'title round-trip' }
        if (($parsed.draft -ne $true) -or ($parsed.draft -isnot [bool])) { $ok = $false; $why = 'typed bool' }
        if ($parsed.labels.Count -ne 2 -or $parsed.labels[0] -cne 'task' -or $parsed.labels[1] -cne 'status:ready') {
            $ok = $false; $why = 'labels array'
        }
        if ($json -notlike '*"draft":true*') { $ok = $false; $why = 'raw typed token' }
        $v = Get-OptixGhArgv -Endpoint 'repos/o/r/issues' -Method 'POST' -Jq '' -InputFile $inFile
        $want = @('api', 'repos/o/r/issues', '-X', 'POST', '--input', $inFile)
        if (($v -join "`0") -cne ($want -join "`0")) { $ok = $false; $why = 'argv vector' }
        foreach ($e in $v) {
            if ($e.Contains('"')) { $ok = $false; $why = 'dquote in argv' }
        }
        $flat = ($v -join ' ')
        foreach ($line in ($case.Text -split "`r?`n")) {
            if ($line.Length -gt 3 -and $flat.Contains($line)) { $ok = $false; $why = 'user text in argv' }
        }
        Remove-Item $inFile, $copyFile -Force -ErrorAction SilentlyContinue
        if ($ok) { Write-Output ("PASS: {0}" -f $case.Name) }
        else { Write-Output ("FAIL: {0} ({1})" -f $case.Name, $why); $fail++ }
    }
} finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
if ($fail -gt 0) { exit 1 }
Write-Output 'All --input cases pass: exact bytes, exact argv, exact JSON.'
