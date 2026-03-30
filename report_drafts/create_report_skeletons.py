from __future__ import annotations

import datetime as _dt
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report_skeletons"


CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


PACKAGE_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


DOCUMENT_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""


STYLES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
        <w:sz w:val="22"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault/>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:after="120" w:line="300" w:lineRule="auto"/>
    </w:pPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="800" w:after="200"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:sz w:val="38"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:after="160"/>
    </w:pPr>
    <w:rPr>
      <w:i/>
      <w:sz w:val="24"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="CoverMeta">
    <w:name w:val="CoverMeta"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:after="80"/>
    </w:pPr>
    <w:rPr>
      <w:sz w:val="22"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="240" w:after="80"/>
      <w:keepNext/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:sz w:val="28"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="160" w:after="40"/>
      <w:keepNext/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:sz w:val="24"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="120" w:after="20"/>
      <w:keepNext/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:sz w:val="22"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Compact">
    <w:name w:val="Compact"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Compact"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:after="40" w:line="260" w:lineRule="auto"/>
    </w:pPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Caption">
    <w:name w:val="Caption"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:i/>
      <w:sz w:val="20"/>
    </w:rPr>
  </w:style>
</w:styles>
"""


APP_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenAI Codex</Application>
</Properties>
"""


def _core_xml(title: str) -> str:
    timestamp = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{escape(title)}</dc:title>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>
"""


def _run(text: str, bold: bool = False, italic: bool = False) -> str:
    # Build WordprocessingML runs directly so this script can emit a minimal
    # .docx package without depending on python-docx in the repo.
    safe = escape(text)
    rpr = []
    if bold:
        rpr.append("<w:b/>")
    if italic:
        rpr.append("<w:i/>")
    rpr_xml = "<w:rPr>%s</w:rPr>" % "".join(rpr) if rpr else ""
    return "<w:r>%s<w:t xml:space=\"preserve\">%s</w:t></w:r>" % (rpr_xml, safe)


def _paragraph(text: str = "", style: str = "Normal", align: str | None = None) -> str:
    if not text:
        return "<w:p/>"
    ppr = ["<w:pStyle w:val=\"%s\"/>" % escape(style)]
    if align:
        ppr.append("<w:jc w:val=\"%s\"/>" % escape(align))
    return (
        "<w:p>"
        "<w:pPr>%s</w:pPr>"
        "%s"
        "</w:p>"
    ) % ("".join(ppr), _run(text))


def _table_cell(text: str, width: int | None = None, header: bool = False) -> str:
    tcpr = []
    if width is not None:
        tcpr.append("<w:tcW w:w=\"%d\" w:type=\"dxa\"/>" % int(width))
    if header:
        tcpr.append("<w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"D9E2F3\"/>")
    style = "Compact"
    align = "center" if header else None
    body = _paragraph(text, style=style, align=align)
    return "<w:tc><w:tcPr>%s</w:tcPr>%s</w:tc>" % ("".join(tcpr), body)


def _table(rows: list[list[str]], widths: list[int] | None = None, header: bool = False) -> str:
    if not rows:
        return ""
    col_count = max(len(row) for row in rows)
    widths = list(widths) if widths else [2400] * col_count
    if len(widths) < col_count:
        widths.extend([widths[-1] if widths else 2400] * (col_count - len(widths)))
    grid = "".join("<w:gridCol w:w=\"%d\"/>" % int(widths[idx]) for idx in range(col_count))
    trs = []
    for row_idx, row in enumerate(rows):
        cells = []
        for idx in range(col_count):
            text = row[idx] if idx < len(row) else ""
            cells.append(_table_cell(text, widths[idx], header=header and row_idx == 0))
        trs.append("<w:tr>%s</w:tr>" % "".join(cells))
    borders = (
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"A6A6A6\"/>"
        "<w:left w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"A6A6A6\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"A6A6A6\"/>"
        "<w:right w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"A6A6A6\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"C9C9C9\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"C9C9C9\"/>"
        "</w:tblBorders>"
    )
    return (
        "<w:tbl>"
        "<w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>%s</w:tblPr>"
        "<w:tblGrid>%s</w:tblGrid>"
        "%s"
        "</w:tbl>"
    ) % (borders, grid, "".join(trs))


def _page_break() -> str:
    return "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"


def _document_xml(paragraphs: list[tuple[str, str]]) -> str:
    body_parts = []
    for kind, value in paragraphs:
        if kind == "page_break":
            body_parts.append(_page_break())
        elif kind == "table":
            rows = value.get("rows", [])
            widths = value.get("widths")
            header = bool(value.get("header", False))
            body_parts.append(_table(rows, widths=widths, header=header))
        else:
            if isinstance(value, dict):
                body_parts.append(
                    _paragraph(
                        value.get("text", ""),
                        style=value.get("style", "Normal"),
                        align=value.get("align"),
                    )
                )
            else:
                text, style = value.split("\u241f", 1)
                body_parts.append(_paragraph(text, style))
    sect = (
        "<w:sectPr>"
        "<w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
        "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
        "<w:body>"
        + "".join(body_parts)
        + sect
        + "</w:body></w:document>"
    )


def _docx_write(path: Path, title: str, paragraphs: list[tuple[str, str]]) -> None:
    # Write the small OOXML package directly because these skeleton files only
    # need stable headings, tables, and metadata placeholders.
    document_xml = _document_xml(paragraphs)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
        zf.writestr("_rels/.rels", PACKAGE_RELS_XML)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", DOCUMENT_RELS_XML)
        zf.writestr("word/styles.xml", STYLES_XML)
        zf.writestr("docProps/core.xml", _core_xml(title))
        zf.writestr("docProps/app.xml", APP_XML)


def _p(text: str, style: str = "Normal") -> tuple[str, str]:
    return ("paragraph", text + "\u241f" + style)


def _break() -> tuple[str, str]:
    return ("page_break", "")


def _t(rows: list[list[str]], widths: list[int] | None = None, header: bool = False) -> tuple[str, dict]:
    return ("table", {"rows": rows, "widths": widths, "header": header})


def _final_report_outline() -> list[tuple[str, str]]:
    return [
        _p("[Project Title]", "Title"),
        _p("Final Design Report", "Subtitle"),
        _p("Course: ENEL 500"),
        _p("Team: [Names]"),
        _p("Sponsor / Supervisor: [Names]"),
        _p("Department / University: [Fill in]"),
        _p("Date: [Fill in]"),
        _p("Version: Draft skeleton"),
        _break(),
        _p("Table of Contents", "Heading1"),
        _p("[Update this in Word after headings are finalized.]"),
        _break(),
        _p("Glossary", "Heading1"),
        _p("[Define key terms, abbreviations, hardware names, model names, and metric names.]"),
        _break(),
        _p("AI Use Disclosure and Verification", "Heading1"),
        _p("[List all AI tools used, versions, what they were used for, whether project data was exposed, and what human verification was performed.]"),
        _p("[Summarize the AI log and note where the full log is stored.]"),
        _p("Executive Summary", "Heading1"),
        _p("[1-2 pages max. State the problem, approach, final system, headline results, and main conclusions.]"),
        _p("Motivation and Objectives", "Heading1"),
        _p("[Explain the engineering problem, who it matters to, and the concrete project objectives.]"),
        _p("[List measurable objectives and map them to sponsor / capstone goals.]"),
        _p("Methodology and Design", "Heading1"),
        _p("System Overview", "Heading2"),
        _p("[Insert one high-level architecture figure and a short walkthrough of the end-to-end pipeline.]"),
        _p("Data Collection Pipeline", "Heading2"),
        _p("[Describe Delsys GUI collection, standard protocol, neutral-recovery protocol, labels, and saved metadata.]"),
        _p("Strict Sensor Placement Workflow", "Heading2"),
        _p("[Explain fixed pair-to-slot mapping, arm-specific channel counts, and fail-closed behavior.]"),
        _p("Preprocessing Pipeline", "Heading2"),
        _p("[Describe resampling, filtering, recalibration, and label generation.]"),
        _p("Model Architecture and Training", "Heading2"),
        _p("[Describe the CNN, input representation, per-subject / cross-subject training, and bundle outputs.]"),
        _p("Realtime Inference and Tuning", "Heading2"),
        _p("[Describe confidence gating, smoothing, hysteresis, ambiguity rejection, and current runtime preset.]"),
        _p("CARLA Integration", "Heading2"),
        _p("[Describe gesture-to-control mapping, scenario setup, logging, and scenario completion rules.]"),
        _p("Product Scope and Functionality", "Heading1"),
        _p("[State the final delivered scope, what works, what is partial, and any deviations from the original plan.]"),
        _p("[Include a short feature checklist or demo-ready capability summary.]"),
        _p("Technical Specifications of Final Product", "Heading1"),
        _p("[List measured technical specs, not just intended specs.]"),
        _p("[Include hardware, software stack, data roots, model variants, active runtime preset, and simulator setup.]"),
        _p("Measures of Success and Validation Tests", "Heading1"),
        _p("Success Criteria", "Heading2"),
        _p("[State the criteria used to judge project success.]"),
        _p("Offline Model Metrics", "Heading2"),
        _p("[Insert offline results table: balanced accuracy, macro F1, accuracy, weighted F1, per-class recall, worst-class recall, confusion-to-neutral rate, neutral prediction false-positive rate.]"),
        _p("Latency Metrics", "Heading2"),
        _p("[Insert latency table: classifier, publish, control, and end-to-end latency with mean / median / p90 / p95 / max.]"),
        _p("CARLA Scenario Metrics", "Heading2"),
        _p("[Insert CARLA results table: mean lane error, lane error RMSE, lane invasions, scenario success / failure, scenario completion time.]"),
        _p("Validation Discussion", "Heading2"),
        _p("[Interpret the results, note pass/fail against objectives, and explain any weak areas.]"),
        _p("Tools, Materials, Supplies, and Cost Analysis", "Heading1"),
        _p("[List hardware, software, simulator resources, and actual vs estimated costs from the Fall term.]"),
        _p("[Explain major cost deviations and whether they were justified.]"),
        _p("Reflection on Societal and Environmental Impact", "Heading1"),
        _p("[Discuss accessibility, safety, misuse risk, privacy, sustainability, and tradeoffs.]"),
        _p("Reflection on Project Management Approach", "Heading1"),
        _p("[Discuss planning, scheduling, risk management, communication, and lessons learned.]"),
        _p("Reflection on Economics and Cost Control", "Heading1"),
        _p("[Discuss budget control, cost-benefit tradeoffs, and resource decisions.]"),
        _p("Reflection on Life-Long Learning and Professional Growth", "Heading1"),
        _p("[Discuss skill development, technical growth, and what changed in your engineering approach.]"),
        _p("Conclusion and Recommendations", "Heading1"),
        _p("[Summarize final outcomes, limitations, and the strongest next steps.]"),
        _p("References", "Heading1"),
        _p("[Insert citations in the required style.]"),
        _p("Appendix A. Full Metric Tables", "Heading1"),
        _p("[Optional full results, extra confusion matrices, or additional scenario outputs.]"),
        _p("Appendix B. AI Use Log Summary", "Heading1"),
        _p("[Paste or summarize the AI log used to satisfy the ENEL 500 disclosure requirement.]"),
        _p("Appendix C. Extra Figures", "Heading1"),
        _p("[System diagrams, sensor placement figures, scenario screenshots, and supplemental charts.]"),
    ]


def _research_paper_outline() -> list[tuple[str, str]]:
    return [
        _p("[Paper Title]", "Title"),
        _p("[Author names and affiliations]", "Subtitle"),
        _p("[Corresponding author / email if needed]"),
        _p("Abstract", "Heading1"),
        _p("[One paragraph: problem, method, dataset / setup, main results, and conclusion.]"),
        _p("Keywords", "Heading1"),
        _p("[EMG, human-machine interface, driving, gesture classification, CNN, CARLA]"),
        _p("1. Introduction", "Heading1"),
        _p("[State the problem, why EMG-based driving interfaces matter, and the gap this work addresses.]"),
        _p("[Summarize the main contribution of this capstone continuation in 3-5 bullets or short paragraphs.]"),
        _p("2. Related Work", "Heading1"),
        _p("[Position this work relative to Basnet et al. (2025) and other EMG / driving-control papers.]"),
        _p("[State clearly what is continued from prior work and what is new here.]"),
        _p("3. System Overview", "Heading1"),
        _p("[Insert a full pipeline figure and one paragraph describing collection to deployment.]"),
        _p("4. Methods", "Heading1"),
        _p("4.1 Data Acquisition", "Heading2"),
        _p("[Describe Delsys-based acquisition, collected labels, standard protocol, and any recovery protocol used.]"),
        _p("4.2 Strict Sensor-Placement Policy", "Heading2"),
        _p("[Describe fixed pair identity, arm-specific channel counts, and session rejection on layout mismatch.]"),
        _p("4.3 Preprocessing", "Heading2"),
        _p("[Describe resampling to 2000 Hz, notch and bandpass filters, and any recalibration behavior.]"),
        _p("4.4 Windowing and Label Generation", "Heading2"),
        _p("[Describe window size, step size, purity threshold, and gesture subset used in reported experiments.]"),
        _p("4.5 Model Architecture", "Heading2"),
        _p("[Describe the residual 1D CNN, channel attention, energy bypass, and normalization behavior.]"),
        _p("4.6 Training Configuration", "Heading2"),
        _p("[Describe per-subject or cross-subject setup, split strategy, augmentation, optimizer, and early stopping / checkpoints if applicable.]"),
        _p("4.7 Deployment and Runtime Configuration", "Heading2"),
        _p("[Describe the reported runtime preset only if it matches the experiments: smoothing, confidence gate, hysteresis, ambiguity rejection, CARLA dwell.]"),
        _p("4.8 Evaluation Protocol", "Heading2"),
        _p("[Describe the final metric set used in this paper: offline metrics, latency metrics, and CARLA scenario metrics.]"),
        _p("5. Experimental Setup", "Heading1"),
        _p("5.1 Hardware and Software", "Heading2"),
        _p("[List Delsys hardware, workstation details, Python / PyTorch environment, and CARLA version.]"),
        _p("5.2 Dataset Scope", "Heading2"),
        _p("[State subjects, arms, sessions, strict roots used, and whether the paper focuses on the 3-gesture subset.]"),
        _p("5.3 CARLA Scenarios", "Heading2"),
        _p("[Describe the lane-keeping course and overtake scenario, including start / finish logic and completion conditions.]"),
        _p("6. Results", "Heading1"),
        _p("6.1 Offline Classification Results", "Heading2"),
        _p("[Insert main offline table and confusion matrix figure.]"),
        _p("6.2 Latency Results", "Heading2"),
        _p("[Insert latency table and, if useful, one latency distribution figure.]"),
        _p("6.3 CARLA Scenario Results", "Heading2"),
        _p("[Insert CARLA metrics table for the reported scenarios.]"),
        _p("6.4 Comparison to Prior Work", "Heading2"),
        _p("[Compare only on metrics that are genuinely comparable to the prior paper.]"),
        _p("7. Discussion", "Heading1"),
        _p("[Interpret what the results mean, especially around neutral flicker, stability, and practical control tradeoffs.]"),
        _p("[State limitations clearly: subject count, scenario scope, gesture subset, or simulator-only constraints.]"),
        _p("8. Conclusion", "Heading1"),
        _p("[Summarize the main outcome and the clearest next technical step.]"),
        _p("Acknowledgments", "Heading1"),
        _p("[Sponsor, lab, supervisors, and any support acknowledgments.]"),
        _p("References", "Heading1"),
        _p("[Insert bibliography in the target style.]"),
        _p("Appendix (Optional)", "Heading1"),
        _p("[Use only if the venue allows it: extra tables, scenario details, or supplementary figures.]"),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_docx = OUT_DIR / "final_report_skeleton.docx"
    paper_docx = OUT_DIR / "research_paper_skeleton.docx"
    _docx_write(final_docx, "Final Design Report Skeleton", _final_report_outline())
    _docx_write(paper_docx, "Research Paper Skeleton", _research_paper_outline())
    print(final_docx)
    print(paper_docx)


if __name__ == "__main__":
    main()
