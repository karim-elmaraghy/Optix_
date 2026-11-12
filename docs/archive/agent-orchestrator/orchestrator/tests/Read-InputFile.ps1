# Test stub: copies a file byte-for-byte (simulates gh reading --input).
# No param block: -File argument binding against a param block fails across
# language modes, so plain positional paths arrive via $args instead.
$ErrorActionPreference = 'Stop'
$InFile = $args[0]
$OutFile = $args[1]
[IO.File]::WriteAllBytes($OutFile, [IO.File]::ReadAllBytes($InFile))
