param(
    [string]$ProjectRoot = "",
    [string]$ProjectKey = "sub-translator-py",
    [string]$ProjectVersion = "",
    [string]$SonarUrl = "",
    [int]$TimeoutSeconds = 600,
    [double]$MinimumCoverage = 65.0,
    [switch]$SkipScanner,
    [switch]$RequireClean
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false, $true)

function Import-LocalSonarEnvironment {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return
    }
    foreach ($line in [System.IO.File]::ReadAllLines($Path, $Utf8NoBom)) {
        if ($line -notmatch '^\s*(SONAR_TOKEN|SONAR_HOST_URL)\s*=\s*(.*)$') {
            continue
        }
        $name = $Matches[1]
        if (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, "Process"))) {
            continue
        }
        $value = $Matches[2].Trim()
        if ($value.Length -ge 2 -and $value[0] -eq $value[$value.Length - 1] -and $value[0] -in @('"', "'")) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

function Get-SonarHeaders {
    $token = [Environment]::GetEnvironmentVariable("SONAR_TOKEN", "Process")
    if ([string]::IsNullOrWhiteSpace($token)) {
        throw "SONAR_TOKEN не задан в окружении процесса или локальном .env."
    }
    $encoded = [Convert]::ToBase64String([System.Text.Encoding]::ASCII.GetBytes($token + ':'))
    return @{ Authorization = "Basic $encoded" }
}

function Invoke-SonarApi {
    param([Parameter(Mandatory = $true)][string]$Path)

    return Invoke-RestMethod -Method Get -Uri ($script:EffectiveSonarUrl + $Path) -Headers $script:Headers
}

function Read-ReportTask {
    param([Parameter(Mandatory = $true)][string]$Path)

    $values = @{}
    foreach ($line in [System.IO.File]::ReadAllLines($Path, $Utf8NoBom)) {
        if ($line -match '^([^=]+)=(.*)$') {
            $values[$Matches[1]] = $Matches[2]
        }
    }
    if ([string]::IsNullOrWhiteSpace($values.ceTaskId)) {
        throw "В report-task.txt отсутствует ceTaskId."
    }
    return $values
}

function Wait-SonarTask {
    param([Parameter(Mandatory = $true)][string]$TaskId)

    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $task = (Invoke-SonarApi -Path "/api/ce/task?id=$TaskId").task
        if ($task.status -in @("SUCCESS", "FAILED", "CANCELED")) {
            return $task
        }
        Start-Sleep -Seconds 1
    } while ([DateTimeOffset]::UtcNow -lt $deadline)
    throw "Задача SonarQube $TaskId не завершилась за $TimeoutSeconds секунд."
}

function Get-MeasureMap {
    param([Parameter(Mandatory = $true)]$Response)

    $values = @{}
    foreach ($measure in $Response.component.measures) {
        $values[$measure.metric] = [double]::Parse(
            [string]$measure.value,
            [System.Globalization.CultureInfo]::InvariantCulture
        )
    }
    return $values
}

function Get-ApplicationVersion {
    param([Parameter(Mandatory = $true)][string]$PythonPath)

    $version = & $PythonPath -B -c "from sub_translate import __version__; print(__version__)"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($version)) {
        throw "Не удалось получить версию приложения из sub_translate.__version__."
    }
    return ([string]$version).Trim()
}

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
Import-LocalSonarEnvironment -Path (Join-Path $ProjectRoot ".env")

if ([string]::IsNullOrWhiteSpace($SonarUrl)) {
    $SonarUrl = [Environment]::GetEnvironmentVariable("SONAR_HOST_URL", "Process")
}
if ([string]::IsNullOrWhiteSpace($SonarUrl)) {
    $SonarUrl = "http://localhost:31339"
}
$script:EffectiveSonarUrl = $SonarUrl.TrimEnd('/')
$script:Headers = Get-SonarHeaders

Set-Location -LiteralPath $ProjectRoot
$pythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python виртуального окружения не найден: $pythonPath"
}
if ([string]::IsNullOrWhiteSpace($ProjectVersion)) {
    $ProjectVersion = Get-ApplicationVersion -PythonPath $pythonPath
}
$coverageGenerated = $false
if (-not $SkipScanner) {
    & $pythonPath -B -m pytest -p no:cacheprovider -q `
        --cov=sub_translate --cov-branch --cov-report=term-missing --cov-report=xml:coverage.xml
    if ($LASTEXITCODE -ne 0) {
        throw "pytest с покрытием завершился с кодом $LASTEXITCODE."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "coverage.xml") -PathType Leaf)) {
        throw "pytest-cov не создал coverage.xml."
    }
    $coverageGenerated = $true
    if (-not (Get-Command sonar-scanner -ErrorAction SilentlyContinue)) {
        throw "sonar-scanner не найден в PATH."
    }
    & sonar-scanner `
        "-Dsonar.host.url=$script:EffectiveSonarUrl" `
        "-Dsonar.projectVersion=$ProjectVersion"
    if ($LASTEXITCODE -ne 0) {
        throw "sonar-scanner завершился с кодом $LASTEXITCODE."
    }
}

$report = Read-ReportTask -Path (Join-Path $ProjectRoot ".scannerwork\report-task.txt")
$task = Wait-SonarTask -TaskId $report.ceTaskId
if ($task.status -ne "SUCCESS") {
    throw "Задача SonarQube $($task.id) завершилась со статусом $($task.status): $($task.errorMessage)"
}

$issues = Invoke-SonarApi -Path "/api/issues/search?componentKeys=$ProjectKey&resolved=false&ps=1"
$security = Invoke-SonarApi -Path (
    "/api/issues/search?componentKeys=$ProjectKey&resolved=false&impactSoftwareQualities=SECURITY&ps=1"
)
$metricKeys = "bugs,code_smells,violations,vulnerabilities,duplicated_lines,duplicated_blocks," +
    "duplicated_lines_density,ncloc,coverage"
$measureResponse = Invoke-SonarApi -Path "/api/measures/component?component=$ProjectKey&metricKeys=$metricKeys"
$measures = Get-MeasureMap -Response $measureResponse
$gate = (Invoke-SonarApi -Path "/api/qualitygates/project_status?projectKey=$ProjectKey").projectStatus.status

$result = [ordered]@{
    projectKey = $ProjectKey
    projectVersion = $ProjectVersion
    scannerExecuted = -not $SkipScanner
    coverageGenerated = $coverageGenerated
    minimumCoverage = $MinimumCoverage
    ceTaskId = $task.id
    ceStatus = $task.status
    qualityGate = $gate
    openIssues = [int]$issues.total
    securityIssues = [int]$security.total
    measures = $measures
}
$result | ConvertTo-Json -Depth 4

if ($RequireClean) {
    $blockingMetrics = @(
        "bugs", "code_smells", "violations", "vulnerabilities",
        "duplicated_lines", "duplicated_blocks", "duplicated_lines_density"
    )
    $dirtyMetric = $blockingMetrics | Where-Object {
        $measures.ContainsKey($_) -and [double]$measures[$_] -ne 0
    }
    $coverage = if ($measures.ContainsKey("coverage")) {
        [double]$measures.coverage
    } else {
        0.0
    }
    $isClean = $gate -eq "OK" -and $issues.total -eq 0 -and $security.total -eq 0 -and
        -not $dirtyMetric -and $coverage -ge $MinimumCoverage
    if (-not $isClean) {
        exit 3
    }
}
