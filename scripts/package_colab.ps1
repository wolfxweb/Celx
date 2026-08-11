param(
    [string]$OutputPath = "dist/Celx-colab.zip"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$destination = Join-Path $projectRoot $OutputPath
$destinationDirectory = Split-Path -Parent $destination

New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null

# Empacota a raiz ativa. O histórico em arquivos/ não entra no ZIP.
$included = @(
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "configs",
    "dataset/benchmark",
    "dataset/curated",
    "dataset/examples",
    "dataset/sql/README.md",
    "docs",
    "legacy_doc",
    "notebooks",
    "scripts",
    "tests"
)

$existingItems = @($included | Where-Object {
    Test-Path -LiteralPath (Join-Path $projectRoot $_)
})

if (Test-Path -LiteralPath $destination) {
    Remove-Item -LiteralPath $destination -Force
}

# O Compress-Archive grava barras invertidas e remove alguns diretórios-raiz no Windows.
# O bsdtar preserva a hierarquia e produz caminhos ZIP portáveis para Colab/Linux.
Push-Location $projectRoot
try {
    & tar.exe -a -c -f $destination `
        --exclude="__pycache__" `
        --exclude="*.pyc" `
        --exclude=".pytest_cache" `
        @existingItems
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao criar o ZIP portátil (código $LASTEXITCODE)."
    }
}
finally {
    Pop-Location
}
Write-Output $destination
