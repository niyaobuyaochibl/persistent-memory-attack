# Run the OpenRouter model suite locally.
#   .\run_api.ps1 pilot      20 users, seed 42, both datasets
#   .\run_api.ps1 full       200 users x 3 seeds
#   .\run_api.ps1 pilot qwen38_27b   restrict to one model tag
# Reads OPENROUTER_API_KEY from the environment.
param([string]$Phase = "pilot", [string]$Only = "")
$ErrorActionPreference = "Stop"
$Code = Split-Path (Split-Path $PSScriptRoot)
if (-not $env:OPENROUTER_API_KEY) { throw "Set OPENROUTER_API_KEY first." }
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"
Set-Location "$Code\agentic_popularity_lockin"
$Root = ($Code -replace '\\', '/')
# tag = model id, pinned provider, required precision ("" = provider default)
$Models = [ordered]@{
    "haiku55"    = @("anthropic/claude-haiku-5.5", "", "")
    "dsv41flash" = @("deepseek/deepseek-v4.1-flash", "deepinfra", "fp8")
    "qwen38_27b" = @("qwen/qwen3.8-27b", "deepinfra", "bf16")
}
foreach ($tag in $Models.Keys) {
    $m = $Models[$tag]
    python -X utf8 scripts/make_revision_configs.py --set api --tag $tag --served-model $m[0] `
        --model-name $m[0] "--provider=$($m[1])" "--quantization=$($m[2])" --root $Root --out configs/api
}
$Filter = if ($Only) { @("--only", $Only) } else { @() }
if ($Phase -eq "pilot") {
    $Seed = if ($Only) { @("--only", "${Only}_main_ml1m_seed42", "${Only}_main_cds_seed42") } else { @("--only", "seed42") }
    python -X utf8 scripts/run_revision_suite.py --backend openai --configs configs/api @Seed --n-users 20
} else {
    python -X utf8 scripts/run_revision_suite.py --backend openai --configs configs/api @Filter
}
