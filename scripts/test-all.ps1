[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [switch]$SkipFrontendBuild
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

function Run-Checked([string]$Label, [scriptblock]$Command) {
    Write-Host $Label -ForegroundColor Green
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE." }
}

# Windows PowerShell 5.1 decodes a .ps1 as the machine's ANSI codepage unless the file carries a
# UTF-8 BOM. A single em dash then arrives as three bytes, one of which is a double quote -- which
# silently terminates whatever string it sits in and reports a parse error somewhere else entirely.
# It cost a whole debugging round to find once. Keeping every launcher pure ASCII removes the
# dependency on the codepage, and this check is cheaper than rediscovering it.
function Assert-AsciiScripts {
    Write-Host 'Checking launcher scripts are pure ASCII' -ForegroundColor Green
    $offenders = @()
    # Launchers live at the repository root and under scripts/. Enumerating those two locations
    # directly is both faster and safer than recursively entering model, package, build and test
    # caches before filtering their results. Some cache directories are intentionally owned by a
    # worker account and are unreadable to the interactive build account.
    $launcherFiles = @(
        Get-ChildItem -LiteralPath $ProjectRoot -File -ErrorAction Stop |
            Where-Object { $_.Extension -in '.ps1', '.bat', '.vbs' }
        Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'scripts') -File -Recurse -ErrorAction Stop |
            Where-Object { $_.Extension -in '.ps1', '.bat', '.vbs' }
    )
    foreach ($file in $launcherFiles) {
        $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
        if ($bytes | Where-Object { $_ -gt 127 }) {
            $offenders += $file.FullName.Substring($ProjectRoot.Length + 1)
        }
    }
    if ($offenders.Count) {
        throw ("These launcher scripts contain non-ASCII bytes and will misparse under Windows " +
               "PowerShell without a BOM:`n  " + ($offenders -join "`n  "))
    }
}

# Every script has to at least parse. A syntax error in a setup script is otherwise only found by
# the person running it, which is the worst possible place to find one.
function Assert-ScriptsParse {
    Write-Host 'Parsing every PowerShell script' -ForegroundColor Green
    $failed = @()
    foreach ($file in (Get-ChildItem -Path (Join-Path $ProjectRoot 'scripts') -Filter *.ps1 -File)) {
        $errors = $null
        [void][System.Management.Automation.Language.Parser]::ParseFile($file.FullName, [ref]$null, [ref]$errors)
        if ($errors -and $errors.Count) {
            $failed += ($file.Name + ' line ' + $errors[0].Extent.StartLineNumber + ': ' + $errors[0].Message)
        }
    }
    if ($failed.Count) { throw ("PowerShell parse errors:`n  " + ($failed -join "`n  ")) }
}

Push-Location -LiteralPath $ProjectRoot
try {
    Assert-AsciiScripts
    Assert-ScriptsParse
    if (-not $SkipInstall) {
        Run-Checked 'Restoring locked JavaScript dependencies' { npm.cmd ci }
    }
    # `npm run check` is `tsc -b`, which builds the referenced projects. Do NOT substitute
    # `tsc --noEmit -p tsconfig.json`: that root config is `"files": []` plus references, so it
    # type-checks exactly nothing and exits 0 no matter how broken the source is.
    Run-Checked 'Type-checking the interface' { npm.cmd run check }
    Run-Checked 'Running interface unit tests' { npm.cmd run test:unit }
    if (-not $SkipFrontendBuild) {
        Run-Checked 'Bundling the interface' { npm.cmd run build }
    }
    Run-Checked 'Running Rust unit and FFmpeg integration tests' { cargo test --manifest-path src-tauri\Cargo.toml }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw 'The project Python environment is missing. Run scripts\bootstrap.ps1 first.'
    }
    $PytestScratch = Join-Path $ProjectRoot 'cache\pytest'
    # -rs names every skipped test and its reason. Without it a skip is one 's' among the dots, and
    # a regression that quietly stops running looks exactly like one that keeps passing. The video
    # writer's round-trip test needs imageio, which lives in the worker venv and not this one, so
    # it skips here; that has to be visible in the log rather than discovered the next time the bug
    # comes back.
    Run-Checked 'Running Python unit and media acceptance tests' { & $Python -m pytest ai-runtime\tests -q -rs --basetemp $PytestScratch }
} finally {
    Pop-Location
}

Write-Host 'All qualification tests passed.' -ForegroundColor Cyan
