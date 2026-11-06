# Full GPT runs on the ChatGPT plan for one model tag; waits out plan quota limits and resumes.
#   .\run_plan_full.ps1 gpt61sol          all seeds
#   .\run_plan_full.ps1 gpt61sol 123      one seed, both datasets
param([Parameter(Mandatory)][string]$Tag, [string]$Seed = "")
$Code = Split-Path (Split-Path $PSScriptRoot)
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"
Set-Location "$Code\agentic_popularity_lockin"
$Only = if ($Seed) { @("${Tag}_main_cds_seed$Seed", "${Tag}_main_ml1m_seed$Seed") } else { @("${Tag}_main_") }
while ($true) {
    python -X utf8 scripts/run_revision_suite.py --backend chatgpt_plan --configs configs/plan --only @Only
    if ($LASTEXITCODE -ne 3) { break }
    Write-Output "$(Get-Date -Format s) quota exhausted; retrying in 30 min"
    Start-Sleep 1800
}
