"""
Department-aware prompt builder for the QA analysis system.

Design philosophy (research-backed):
- Two-phase prompting: (1) evidence gathering via chain-of-thought, (2) structured fill.
  Based on Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
  — forcing the model to surface evidence quotes BEFORE classifying significantly reduces
  hallucination and false positives.
- Explicit enumeration of PHI categories in the prompt so the model knows exactly what
  constitutes a HIPAA concern (not a vague concept).
- Department-specific rules injected per call — mirrors real QA team workflows where
  Scheduling reviewers check different things than Records reviewers.
- "positive_interaction" flag type prominently mentioned to encourage non-punitive framing.
- "If ambiguous, note it — do NOT assume the worst" is instruction-locked to prevent
  false escalations.
"""

from __future__ import annotations

from app.models.input_models import CallTranscript, Department

# ─────────────────────────────────────────────────────────────────────────────
# Department-Specific Rules
# ─────────────────────────────────────────────────────────────────────────────

_DEPARTMENT_RULES: dict[Department, str] = {
    Department.SCHEDULING: """
SCHEDULING DEPARTMENT RULES:
- MUST CHECK: Was the appointment date AND time confirmed clearly with the caller?
  If not confirmed, flag as protocol_violation (minor severity).
- MUST CHECK: Did the agent offer or confirm a callback number or confirmation method?
- POSITIVE: Agent proactively offered appointment reminders or instructions.
""".strip(),

    Department.ONBOARDING: """
ONBOARDING DEPARTMENT RULES:
- MUST CHECK: Was the lien agreement or financial responsibility discussed or mentioned?
  If the call was substantive and this was omitted, flag as protocol_violation (moderate severity).
- MUST CHECK: Was the patient given clear next steps for their onboarding process?
- POSITIVE: Agent explained the process clearly and set accurate expectations.
""".strip(),

    Department.HELPDESK: """
HELPDESK DEPARTMENT RULES:
- MUST CHECK: Was the caller's issue addressed or escalated to an appropriate resource?
  If left unresolved without explanation or follow-up, flag as protocol_violation (moderate severity).
- MUST CHECK: Did the agent provide a ticket number, reference, or callback commitment?
- POSITIVE: Agent demonstrated problem-solving and ownership of the caller's issue.
""".strip(),

    Department.FOLLOW_UPS: """
FOLLOW-UPS DEPARTMENT RULES:
- MUST CHECK: Was a callback date/time or next contact point established?
  If not, flag as protocol_violation (minor severity).
- MUST CHECK: Were any open action items from the previous interaction addressed?
- POSITIVE: Agent maintained continuity and showed awareness of caller history.
""".strip(),

    Department.RECORDS: """
RECORDS DEPARTMENT RULES:
- MUST CHECK: Was patient authorization or release-of-records consent verified BEFORE
  any medical records or sensitive information was discussed or released?
  If not, this is a CRITICAL hipaa_concern — flag immediately.
- MUST CHECK: Was the identity of the caller verified before sharing information?
  If patient info was given to an unverified caller, this is CRITICAL.
- POSITIVE: Agent properly verified identity and followed release procedures.
""".strip(),
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Construction
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior call quality analyst for a medical practice management company \
that operates virtual assistants across 39+ clinic locations.

Your SOLE job is to analyze a phone call transcript and produce a structured quality assessment.

═══════════════════════════════════════════════════════
CRITICAL OPERATING RULES (NON-NEGOTIABLE)
═══════════════════════════════════════════════════════
1. DO NOT INVENT ISSUES. Only flag what is explicitly present in the transcript.
2. DO NOT ASSUME THE WORST. If something is ambiguous, describe the ambiguity — do not escalate it.
3. SEPARATE FACTS FROM INFERENCES. If you are inferring, say "the agent appears to..." not "the agent did..."
4. "escalate" is reserved for CONFIRMED critical issues ONLY — you must quote the exact transcript evidence.
5. When in doubt between "escalate" and "needs_review", always choose "needs_review".
6. ALWAYS include a positive_interaction flag if the agent did anything well.

═══════════════════════════════════════════════════════
THRESHOLD DEFINITIONS
═══════════════════════════════════════════════════════
"escalate" → Use ONLY if ONE or more of these is CONFIRMED in the transcript:
  • HIPAA violation: PHI (name + DOB, SSN, account#, diagnosis, medication) shared with a clearly
    unauthorized party (e.g., third party who is not the patient, guardian, or authorized representative)
  • Explicit rudeness: Agent uses profanity, insults, dismissive/hostile language toward the caller
  • Dangerous misinformation: Incorrect medical, legal, or financial information that could directly
    harm the caller's health, legal standing, or financial wellbeing

"needs_review" → Use for:
  • Ambiguous PHI disclosure (unclear if caller was authorized)
  • Possible misinformation that needs human verification
  • Protocol gaps or incomplete processes
  • Tone that is cold or unprofessional but not explicitly rude
  • Short calls where limited analysis confidence exists

"pass" → Use when:
  • Call was handled professionally with no significant issues
  • Minor imperfections do not affect quality of service

═══════════════════════════════════════════════════════
HIPAA PHI CATEGORIES (flag as hipaa_concern if shared improperly)
═══════════════════════════════════════════════════════
• Patient full name combined with any other PHI
• Date of birth (DOB) / Social Security Number (SSN)
• Medical record number / account number / insurance ID
• Diagnosis, condition, or treatment information
• Prescription/medication details
• Appointment details combined with identifying information

═══════════════════════════════════════════════════════
SCORING GUIDANCE
═══════════════════════════════════════════════════════
professionalism_score: Tone, empathy, courtesy, active listening. 
  1.0 = warm, empathetic, clear. 0.0 = hostile, dismissive, rude.
accuracy_score: Correctness of information given. If no information given, score neutrally (0.7).
resolution_score: Was the caller's goal met? 
  1.0 = fully resolved. 0.5 = partially addressed. 0.0 = ignored or made worse.

ALWAYS populate strengths and improvements with SPECIFIC evidence from the transcript.
Improvements should be constructive and actionable — not punitive.
""".strip()


def build_user_prompt(call: CallTranscript) -> str:
    """
    Build the user-turn prompt for a specific call.
    Injects department-specific rules and call metadata.

    The two-phase structure asks the model to:
    1. Note evidence for each potential issue category (chain-of-thought anchor)
    2. Produce the structured assessment
    """
    dept_rules = _DEPARTMENT_RULES.get(call.department, "No specific department rules apply.")

    # Duration classification for edge case guidance
    if call.call_duration_seconds < 30:
        duration_note = (
            f"⚠️  NOTE: This call is very short ({call.call_duration_seconds} seconds). "
            "Limited analysis confidence — prefer 'needs_review' over definitive assessments. "
            "Note this limitation in your assessment_reasoning."
        )
    elif call.call_duration_seconds < 120:
        duration_note = (
            f"Note: This is a brief call ({call.call_duration_seconds} seconds). "
            "Consider this when assessing resolution."
        )
    else:
        duration_note = f"Call duration: {call.call_duration_seconds} seconds."

    return f"""════════════════════════════════════════
CALL METADATA
════════════════════════════════════════
Call ID:    {call.call_id}
Agent:      {call.agent_name}
Date:       {call.call_date}
Department: {call.department.value}
{duration_note}

════════════════════════════════════════
DEPARTMENT-SPECIFIC QA RULES
════════════════════════════════════════
{dept_rules}

════════════════════════════════════════
TRANSCRIPT
════════════════════════════════════════
{call.transcript.strip()}

════════════════════════════════════════
ANALYSIS INSTRUCTIONS
════════════════════════════════════════
Before filling the structured output, mentally note:
• Did you see ANY evidence of PHI being shared with an unauthorized party?
• Did you see ANY explicit rudeness or hostility from the agent?
• Did you see ANY clearly incorrect information that could harm the caller?
• What did the agent do WELL? (Required: at least one strength)
• Were the department-specific rules followed?

Now produce the complete structured quality analysis for call_id="{call.call_id}".
Remember: Only flag what you can QUOTE from the transcript above.
""".strip()
