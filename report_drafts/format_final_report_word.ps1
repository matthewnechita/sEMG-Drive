param(
    [string]$InputPath = "report_drafts/final_report_draft_v2.docx",
    [string]$OutputPath = "report_drafts/final_report_draft_v3.docx"
)

$ErrorActionPreference = "Stop"

$inputResolved = (Resolve-Path $InputPath).Path
$outputResolved = Join-Path (Get-Location) $OutputPath
$outputDir = Split-Path $outputResolved -Parent
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$heading1Texts = @(
    "Table of Contents",
    "Glossary",
    "AI Use Disclosure and Verification",
    "Executive Summary",
    "Motivation and Objectives",
    "Methodology and Design",
    "Product Scope and Functionality",
    "Technical Specifications of Final Product",
    "Measures of Success and Validation Tests",
    "Tools, Materials, Supplies, and Cost Analysis",
    "Reflection on Societal and Environmental Impact",
    "Reflection on Project Management Approach",
    "Reflection on Economics and Cost Control",
    "Reflection on Life-Long Learning and Professional Growth",
    "Conclusion and Recommendations",
    "References",
    "Appendix A. Commands Used for Metric Collection",
    "Appendix B. Additional Figures and Tables"
)

$heading2Texts = @(
    "Project Snapshot",
    "System Overview",
    "Data Collection Pipeline",
    "Strict Sensor Placement Workflow",
    "Preprocessing Pipeline",
    "Windowing, Label Generation, and Model Architecture",
    "Training Configuration",
    "Realtime Inference and Tuning",
    "CARLA Integration",
    "Success Criteria",
    "Main Metric Set",
    "Validation Workflow",
    "Results Placeholders",
    "Validation Discussion"
)

$word = $null
$doc = $null

try {
    Copy-Item $inputResolved $outputResolved -Force

    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $doc = $word.Documents.Open($outputResolved)

    $doc.PageSetup.TopMargin = $word.InchesToPoints(1.0)
    $doc.PageSetup.BottomMargin = $word.InchesToPoints(1.0)
    $doc.PageSetup.LeftMargin = $word.InchesToPoints(1.0)
    $doc.PageSetup.RightMargin = $word.InchesToPoints(1.0)

    $doc.Sections.Item(1).PageSetup.DifferentFirstPageHeaderFooter = $true

    $mainFooter = $doc.Sections.Item(1).Footers.Item(1)
    $mainFooter.Range.Text = "ENEL 500 Final Design Report Draft"
    $mainFooter.Range.ParagraphFormat.Alignment = 1
    $null = $mainFooter.PageNumbers.Add()

    $mainHeader = $doc.Sections.Item(1).Headers.Item(1)
    $mainHeader.Range.Text = "[Project Title]"
    $mainHeader.Range.ParagraphFormat.Alignment = 1

    foreach ($paragraph in $doc.Paragraphs) {
        $text = ($paragraph.Range.Text -replace "[`r`a]", "").Trim()
        if ([string]::IsNullOrWhiteSpace($text)) {
            continue
        }
        if ($heading1Texts -contains $text) {
            $paragraph.Range.Style = "Heading 1"
        } elseif ($heading2Texts -contains $text) {
            $paragraph.Range.Style = "Heading 2"
        } elseif ($text -eq "[Project Title]") {
            $paragraph.Range.Style = "Title"
        } elseif ($text -eq "Final Design Report Draft") {
            $paragraph.Range.Style = "Subtitle"
        } elseif ($text -like "Course:*" -or $text -like "Team:*" -or $text -like "Sponsor / Supervisor:*" -or $text -like "Date:*") {
            $paragraph.Range.Style = "Subtitle"
            $paragraph.Range.ParagraphFormat.Alignment = 1
        }
    }

    foreach ($table in $doc.Tables) {
        $table.Style = "Table Grid"
        $table.Rows.Alignment = 1
        $table.Range.ParagraphFormat.SpaceAfter = 4
        $table.Range.ParagraphFormat.SpaceBefore = 0
        $table.AutoFitBehavior(2)
    }

    $doc.Save()
    Write-Output $outputResolved
}
finally {
    if ($doc -ne $null) {
        $doc.Close($false) | Out-Null
    }
    if ($word -ne $null) {
        $word.Quit() | Out-Null
    }
}
