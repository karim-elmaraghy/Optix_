# Optix git helper. Every git call carries an explicit -c safe.directory so the
# dispatcher works regardless of global gitconfig. Never resets, force-pushes,
# or deletes branches/worktrees.

function Invoke-RepoGit {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string[]]$GitArgs,
        [string[]]$ExtraSafe = @()
    )
    $pre = @('-c', ("safe.directory={0}" -f $RepoRoot))
    foreach ($s in $ExtraSafe) { $pre += @('-c', ("safe.directory={0}" -f $s)) }
    $out = & git @pre @GitArgs 2>&1
    if ($LASTEXITCODE -ne 0) { throw ('git {0} failed: {1}' -f ($GitArgs -join ' '), ($out -join "`n")) }
    return $out
}

function Get-OptixRepoRoot {
    param([Parameter(Mandatory = $true)][string]$OrchestratorRoot)
    return (Get-Item (Join-Path $OrchestratorRoot '..\..')).FullName
}
