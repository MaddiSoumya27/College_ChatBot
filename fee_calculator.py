"""
fee_calculator.py
-----------------
Fee calculation tool for BVRIT Hyderabad.
Provides the actual computation function and the tool definition
that can be called by the LLM via function/tool calling.

Fee structure from verified KB:
- Tuition fee: ₹1,20,000 per year for ALL B.Tech programs
- Other fees are estimated based on typical college fee structures
"""

import math
from typing import Optional

# ── Verified fee data from BVRIT Hyderabad Knowledge Base ──────────────────────

TUITION_FEE_PER_YEAR = 120_000  # ₹1,20,000 for all B.Tech programs

# Estimated additional fees (based on typical Telangana engineering college structure)
REGISTRATION_FEE = 2_500       # One-time
CAUTION_DEPOSIT = 5_000        # One-time, refundable
EXAM_FEE_PER_YEAR = 6_000      # Per year
LIBRARY_FEE_PER_YEAR = 3_000   # Per year
SPORTS_FEE_PER_YEAR = 2_000    # Per year
LAB_FEE_PER_YEAR = 4_000       # Per year (for programs with lab-intensive courses)

# Hostel fees (estimated based on typical ranges — contact college for exact)
HOSTEL_FEE_PER_YEAR = 60_000   # ₹60,000 per year (twin-sharing)
HOSTEL_SINGLE_PER_YEAR = 90_000  # ₹90,000 per year (single room)
MESS_FEE_PER_YEAR = 36_000      # ₹36,000 per year (approx ₹3,000/month)

# Transport fee
TRANSPORT_FEE_PER_YEAR = 25_000  # ₹25,000 per year (optional)

# Program names (valid B.Tech programs at BVRIT Hyderabad)
VALID_PROGRAMS = {
    "cse": "CSE – Computer Science and Engineering",
    "computer science": "CSE – Computer Science and Engineering",
    "computer science and engineering": "CSE – Computer Science and Engineering",
    "cse-aiml": "CSE-AIML – CSE Artificial Intelligence & Machine Learning",
    "aiml": "CSE-AIML – CSE Artificial Intelligence & Machine Learning",
    "ai & ml": "CSE-AIML – CSE Artificial Intelligence & Machine Learning",
    "ece": "ECE – Electronics and Communication Engineering",
    "electronics": "ECE – Electronics and Communication Engineering",
    "electronics and communication": "ECE – Electronics and Communication Engineering",
    "eee": "EEE – Electrical and Electronics Engineering",
    "electrical": "EEE – Electrical and Electronics Engineering",
    "it": "IT – Information Technology",
    "information technology": "IT – Information Technology",
    "bs&h": "BS&H – Basic Sciences and Humanities",
    "basic sciences": "BS&H – Basic Sciences and Humanities",
}

# M.Tech programs
VALID_PG_PROGRAMS = {
    "data sciences": "M.Tech Data Sciences (intake: 18)",
    "data science": "M.Tech Data Sciences (intake: 18)",
    "cse pg": "M.Tech Computer Science and Engineering (intake: 12)",
    "computer science pg": "M.Tech Computer Science and Engineering (intake: 12)",
    "vlsi": "M.Tech VLSI Design (intake: 12)",
    "vlsi design": "M.Tech VLSI Design (intake: 12)",
}

M_TECH_TUITION_PER_YEAR = 90_000  # Estimated for M.Tech


def calculate_fee(
    program: str,
    year: int = 1,
    include_hostel: bool = False,
    hostel_type: str = "shared",
    include_transport: bool = False,
    scholarship_pct: float = 0.0,
) -> dict:
    """
    Calculate the fee breakdown for a student at BVRIT Hyderabad.

    Parameters
    ----------
    program : str
        Name of the program (e.g., 'CSE', 'ECE', 'CSE-AIML', etc.)
    year : int
        Year of study (1-4 for B.Tech, 1-2 for M.Tech)
    include_hostel : bool
        Whether to include hostel and mess fees
    hostel_type : str
        'shared' (twin-sharing) or 'single'
    include_transport : bool
        Whether to include transport fee
    scholarship_pct : float
        Scholarship percentage applied on tuition fee only (0–100).
        e.g. 20.0 means 20% off tuition.

    Returns
    -------
    dict with keys: program_name, year, fee_breakdown, total, notes
    """
    # Normalise program name
    prog_lower = program.strip().lower()
    prog_name = VALID_PROGRAMS.get(prog_lower)
    is_pg = False
    if prog_name is None:
        prog_name = VALID_PG_PROGRAMS.get(prog_lower)
        is_pg = True

    # Build breakdown
    breakdown = {}

    # One-time fees (year 1 only)
    if year == 1:
        breakdown["Registration Fee (one-time)"] = REGISTRATION_FEE
        breakdown["Caution Deposit (refundable)"] = CAUTION_DEPOSIT

    # Tuition
    base_tuition = M_TECH_TUITION_PER_YEAR if is_pg else TUITION_FEE_PER_YEAR
    breakdown["Tuition Fee (per year)"] = base_tuition

    # Annual fees
    breakdown["Examination Fee (per year)"] = EXAM_FEE_PER_YEAR
    breakdown["Library Fee (per year)"] = LIBRARY_FEE_PER_YEAR
    breakdown["Sports Fee (per year)"] = SPORTS_FEE_PER_YEAR
    breakdown["Laboratory Fee (per year)"] = LAB_FEE_PER_YEAR

    # Optional fees
    if include_hostel:
        if hostel_type == "single":
            breakdown["Hostel Fee (single room, per year)"] = HOSTEL_SINGLE_PER_YEAR
        else:
            breakdown["Hostel Fee (shared room, per year)"] = HOSTEL_FEE_PER_YEAR
        breakdown["Mess Fee (per year)"] = MESS_FEE_PER_YEAR

    if include_transport:
        breakdown["Transport Fee (per year)"] = TRANSPORT_FEE_PER_YEAR

    total = sum(breakdown.values())

    # Apply scholarship discount on tuition only
    scholarship_amount = 0.0
    scholarship_pct = max(0.0, min(100.0, scholarship_pct))  # clamp to [0, 100]
    if scholarship_pct > 0:
        scholarship_amount = round(base_tuition * scholarship_pct / 100)
        breakdown[f"Scholarship ({scholarship_pct:g}% on Tuition)"] = -int(scholarship_amount)
        total -= int(scholarship_amount)

    # Generate notes
    notes = [
        "• Tuition fee of ₹1,20,000/year applies to ALL B.Tech programs (2022-2025 batches).",
        "• Fees listed above are indicative. Contact BVRIT Hyderabad for the exact current fee structure.",
        "• Hostel fee varies by room type; contact admissions for current rates.",
        "• Scholarships and fee waivers may be available based on merit and category.",
        "• Contact: Dr. J. Manoj Kumar (Admissions) — 92471 64714",
    ]
    if is_pg:
        notes[0] = f"• M.Tech tuition fee is approximately ₹{M_TECH_TUITION_PER_YEAR:,}/year."
    if scholarship_pct > 0:
        notes.insert(0, f"• Scholarship of {scholarship_pct:g}% applied on tuition only. Actual discount: ₹{int(scholarship_amount):,}.")

    result = {
        "program_name": prog_name or program,
        "year": year,
        "is_pg": is_pg,
        "fee_breakdown": breakdown,
        "total": int(total),
        "notes": notes,
        "scholarship_pct": scholarship_pct,
        "scholarship_amount": int(scholarship_amount),
    }

    return result


def calculate_hostel_only(
    hostel_type: str = "shared",
    total_years: int = 4,
) -> dict:
    """
    Return ONLY the hostel + mess cost for 1 year or N years.
    Used when the user asks purely about hostel cost without wanting
    the full tuition/lab/library breakdown.
    """
    h_fee   = HOSTEL_SINGLE_PER_YEAR if hostel_type == "single" else HOSTEL_FEE_PER_YEAR
    h_label = "Single room" if hostel_type == "single" else "Shared room (twin-sharing)"

    if total_years == 1:
        breakdown = {
            f"Hostel Fee ({h_label}, per year)": h_fee,
            "Mess Fee (per year)":               MESS_FEE_PER_YEAR,
        }
        total = h_fee + MESS_FEE_PER_YEAR
        period = "per year"
    else:
        breakdown = {
            f"Hostel Fee ({h_label})  (Rs.{h_fee:,}/yr × {total_years} yrs)": h_fee * total_years,
            f"Mess Fee         (Rs.{MESS_FEE_PER_YEAR:,}/yr × {total_years} yrs)":  MESS_FEE_PER_YEAR * total_years,
        }
        total = (h_fee + MESS_FEE_PER_YEAR) * total_years
        period = f"over {total_years} years"

    notes = [
        f"• Hostel fee ({h_label.lower()}): ₹{h_fee:,}/year.",
        f"• Mess fee: ₹{MESS_FEE_PER_YEAR:,}/year (approx ₹{MESS_FEE_PER_YEAR//12:,}/month).",
        "• Hostel and mess fees are indicative — contact BVRIT Hyderabad for exact current rates.",
        "• Hostel availability is subject to seat availability; apply early.",
        "• Contact: Dr. J. Manoj Kumar (Admissions) — 92471 64714",
    ]

    return {
        "program_name": "Hostel & Mess",
        "year":         total_years,
        "total_years":  total_years,
        "hostel_only":  True,
        "period":       period,
        "fee_breakdown": breakdown,
        "total":        total,
        "notes":        notes,
    }


def calculate_total_course_fee(
    program: str,
    include_hostel: bool = False,
    hostel_type: str = "shared",
    include_transport: bool = False,
    scholarship_pct: float = 0.0,
) -> dict:
    """
    Calculate the TOTAL fee for the complete B.Tech (4 years) or M.Tech (2 years) course.
    One-time fees are counted once. Annual fees are multiplied by number of years.
    Scholarship percentage is applied on tuition only.
    """
    prog_lower = program.strip().lower()
    prog_name = VALID_PROGRAMS.get(prog_lower)
    is_pg = False
    if prog_name is None:
        prog_name = VALID_PG_PROGRAMS.get(prog_lower)
        is_pg = True

    total_years = 2 if is_pg else 4
    tuition = M_TECH_TUITION_PER_YEAR if is_pg else TUITION_FEE_PER_YEAR
    degree = "M.Tech" if is_pg else "B.Tech"

    # One-time fees (charged once for the whole course)
    breakdown = {
        "Registration Fee (one-time)":            REGISTRATION_FEE,
        "Caution Deposit (refundable, one-time)": CAUTION_DEPOSIT,
    }

    # Annual fees × total years
    breakdown[f"Tuition Fee      (Rs.{tuition:,}/yr x {total_years} yrs)"]           = tuition * total_years
    breakdown[f"Examination Fee  (Rs.{EXAM_FEE_PER_YEAR:,}/yr x {total_years} yrs)"] = EXAM_FEE_PER_YEAR * total_years
    breakdown[f"Library Fee      (Rs.{LIBRARY_FEE_PER_YEAR:,}/yr x {total_years} yrs)"]  = LIBRARY_FEE_PER_YEAR * total_years
    breakdown[f"Sports Fee       (Rs.{SPORTS_FEE_PER_YEAR:,}/yr x {total_years} yrs)"]   = SPORTS_FEE_PER_YEAR * total_years
    breakdown[f"Laboratory Fee   (Rs.{LAB_FEE_PER_YEAR:,}/yr x {total_years} yrs)"]      = LAB_FEE_PER_YEAR * total_years

    if include_hostel:
        h_fee   = HOSTEL_SINGLE_PER_YEAR if hostel_type == "single" else HOSTEL_FEE_PER_YEAR
        h_label = "single" if hostel_type == "single" else "shared"
        breakdown[f"Hostel Fee ({h_label})  (Rs.{h_fee:,}/yr x {total_years} yrs)"]      = h_fee * total_years
        breakdown[f"Mess Fee         (Rs.{MESS_FEE_PER_YEAR:,}/yr x {total_years} yrs)"] = MESS_FEE_PER_YEAR * total_years

    if include_transport:
        breakdown[f"Transport Fee    (Rs.{TRANSPORT_FEE_PER_YEAR:,}/yr x {total_years} yrs)"] = TRANSPORT_FEE_PER_YEAR * total_years

    total = sum(breakdown.values())

    # Apply scholarship discount on tuition only
    scholarship_amount = 0
    scholarship_pct = max(0.0, min(100.0, scholarship_pct))  # clamp to [0, 100]
    if scholarship_pct > 0:
        tuition_total = tuition * total_years
        scholarship_amount = round(tuition_total * scholarship_pct / 100)
        breakdown[f"Scholarship ({scholarship_pct:g}% on Tuition, {total_years} yrs)"] = -scholarship_amount
        total -= scholarship_amount

    notes = [
        f"• Total calculated over {total_years} years (complete {degree} program).",
        "• Tuition is Rs.1,20,000/year for all B.Tech programs (2022-2025 batches).",
        "• Caution deposit of Rs.5,000 is refundable at graduation.",
        "• All fees are indicative — contact BVRIT Hyderabad for the exact structure.",
        "• Scholarships/fee reimbursement may reduce total cost based on merit & category.",
        "• Contact: Dr. J. Manoj Kumar (Admissions) — 92471 64714 | info@bvrithyderabad.edu.in",
    ]
    if scholarship_pct > 0:
        notes.insert(0, f"• Scholarship of {scholarship_pct:g}% applied on tuition (₹{tuition:,}/yr × {total_years} yrs = ₹{tuition*total_years:,}). Discount: ₹{scholarship_amount:,}.")

    return {
        "program_name": prog_name or program,
        "year": total_years,
        "total_years": total_years,
        "is_pg": is_pg,
        "multi_year": True,
        "fee_breakdown": breakdown,
        "total": int(total),
        "notes": notes,
        "scholarship_pct": scholarship_pct,
        "scholarship_amount": scholarship_amount,
    }


# ── Tool definition for LLM function calling ───────────────────────────────────

FEE_CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate_fee",
        "description": (
            "Calculate the detailed fee breakdown for a student at BVRIT Hyderabad "
            "College of Engineering for Women. Use this when the user asks about fee "
            "structure, fee breakdown, total fees, hostel fees, or cost calculations "
            "for specific programs and years."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "program": {
                    "type": "string",
                    "description": (
                        "The academic program name. Examples: 'CSE', 'ECE', 'EEE', "
                        "'IT', 'CSE-AIML', 'M.Tech Data Sciences', etc."
                    ),
                },
                "year": {
                    "type": "integer",
                    "description": "Year of study (1-4 for B.Tech, 1-2 for M.Tech). Default is 1.",
                },
                "include_hostel": {
                    "type": "boolean",
                    "description": "Whether to include hostel and mess fees. Default is false.",
                },
                "hostel_type": {
                    "type": "string",
                    "enum": ["shared", "single"],
                    "description": "Hostel room type: 'shared' (twin-sharing) or 'single'. Default is 'shared'.",
                },
                "include_transport": {
                    "type": "boolean",
                    "description": "Whether to include transport/bus fee. Default is false.",
                },
                "scholarship_pct": {
                    "type": "number",
                    "description": (
                        "Scholarship percentage to apply on tuition fee only (0–100). "
                        "For example, 20 means 20% off tuition. Default is 0."
                    ),
                },
            },
            "required": ["program"],
        },
    },
}


def format_fee_result(result: dict) -> str:
    """Format fee result. Handles single-year, multi-year, and hostel-only results."""
    lines = []
    is_multi    = result.get("multi_year", False)
    hostel_only = result.get("hostel_only", False)
    total_years = result.get("total_years", result.get("year", 1))
    degree = "M.Tech" if result.get("is_pg") else "B.Tech"
    scholarship_pct = result.get("scholarship_pct", 0.0)
    scholarship_amount = result.get("scholarship_amount", 0)

    if hostel_only:
        period = result.get("period", "per year")
        lines.append(f"### 🏠 Hostel & Mess Fee — {period.title()}")
    elif is_multi:
        title = f"### 💰 Total {degree} Course Fee — {result['program_name']} ({total_years} Years)"
        if scholarship_pct > 0:
            title += f" | {scholarship_pct:g}% Scholarship Applied"
        lines.append(title)
    else:
        title = f"### 💰 Fee Breakdown — {result['program_name']} (Year {result['year']})"
        if scholarship_pct > 0:
            title += f" | {scholarship_pct:g}% Scholarship Applied"
        lines.append(title)

    lines.append("")
    lines.append("| Fee Component | Amount (₹) |")
    lines.append("|---|---:|")

    for component, amount in result["fee_breakdown"].items():
        if amount < 0:
            # Scholarship discount — render in green-ish with minus sign
            lines.append(f"| 🎓 {component} | **-₹{abs(amount):,}** |")
        else:
            lines.append(f"| {component} | ₹{amount:,} |")

    if hostel_only:
        period = result.get("period", "per year")
        label = f"Total Hostel + Mess ({period})"
    elif is_multi:
        label = f"Total {total_years}-Year Cost"
    else:
        label = "Total (This Year)"

    lines.append(f"| **{label}** | **₹{result['total']:,}** |")
    lines.append("")

    if is_multi and not hostel_only:
        avg = result["total"] // total_years
        lines.append(f"> 💡 **Per-year average: ₹{avg:,}** (one-time fees in Year 1 only)")
        if scholarship_amount > 0:
            lines.append(f"> 🎓 **Scholarship saving: ₹{scholarship_amount:,}** over {total_years} years ({scholarship_pct:g}% on tuition)")
        lines.append("")

    lines.append("**📝 Notes:**")
    for note in result["notes"]:
        lines.append(note)
    lines.append("")
    lines.append("---")
    lines.append(
        "📞 *For exact fee confirmation:* "
        "**Dr. J. Manoj Kumar — 92471 64714** or **info@bvrithyderabad.edu.in**"
    )
    return "\n".join(lines)
