# Run the GPT suite on the ChatGPT plan (no API cost; default sampling, no temperature control).
#   .\run_plan.ps1 pilot      20 users, seed 42, both datasets
#   .\run_plan.ps1 full       200 users x 3 seeds
#   .\run_plan.ps1 pilot gpt6luna   restrict to one model tag
# On exit code 3 the plan quota is exhausted: wait ~30 min and re-run the same command (responses are cached).
param([string]$Phase = "pilot", [string]$Only = "")
$ErrorActionPreference = "Stop"
$Code = Split-Path (Split-Path $PSScriptRoot)
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"
Set-Location "$Code\agentic_popularity_lockin"
$Root = ($Code -replace '\\', '/')
# tag = model slug, reasoning effort
$Models = [ordered]@{
    "gpt6luna"  = @("gpt-6-luna", "none")
    "gpt61sol"  = @("gpt-6.1-sol", "low")
}
foreach ($tag in $Models.Keys) {
    $m = $Models[$tag]
    python -X utf8 scripts/make_revision_configs.py --set plan --tag $tag --served-model $m[0] `
        --model-name $m[0] "--effort=$($m[1])" --root $Root --out configs/plan
}
$Filter = if ($Only) { @("--only", $Only) } else { @() }
if ($Phase -eq "pilot") {
    $Seed = if ($Only) { @("--only", "${Only}_main_ml1m_seed42", "${Only}_main_cds_seed42") } else { @("--only", "seed42") }
    python -X utf8 scripts/run_revision_suite.py --backend chatgpt_plan --configs configs/plan @Seed --n-users 20
} else {
    python -X utf8 scripts/run_revision_suite.py --backend chatgpt_plan --configs configs/plan @Filter
}
exit $LASTEXITCODE
