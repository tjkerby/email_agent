from scenario_loader import load_scenario
from email_agent import EmailAgent, EmailMessage
from grading_serialization import grading_result_to_storage
import json


def _prompt_student_email(reply_subject: str) -> EmailMessage:
    """Collect a multi-line email body from stdin."""

    print("Please type the student's email reply. Finish input with a line containing only '.'")
    print("(press Ctrl+D on UNIX systems if you prefer to end the input early)\n")

    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == ".":
            break
        lines.append(line)

    body = "\n".join(lines).strip()
    if not body:
        raise SystemExit("No student email text provided. Exiting.")

    return EmailMessage(
        sender="Student",
        subject=f"Re: {reply_subject}",
        body=body,
    )


def main():
    # ------------------------------------------------------------
    # 1. Load the scenario configuration (JSON or YAML)
    # ------------------------------------------------------------
    scenario = load_scenario("scenarios/missed_remote_standup.json")

    # Create the email agent for this scenario
    agent = EmailAgent(
        model="gpt-oss:20b",
        temperature=0.2,
        scenario=scenario,
    )

    # ------------------------------------------------------------
    # 2. Build the starter thread (email from manager/client)
    # ------------------------------------------------------------
    starter_thread = agent.build_starter_thread()
    starter_email = starter_thread[0]

    print("\n=== STARTER EMAIL (Manager → Student) ===\n")
    print(f"Subject: {starter_email.subject}")
    print(f"From: {starter_email.sender}\n")
    print(starter_email.body)
    print("\n=====================================================\n")

    # ------------------------------------------------------------
    # 3. Student replies (in production this comes from webhook)
    # ------------------------------------------------------------
    student_email = _prompt_student_email(starter_email.subject)

    # ------------------------------------------------------------
    # 4. Single unified call:
    #    → AI grades the student email
    #    → AI responds as counterpart
    # ------------------------------------------------------------
    result = agent.evaluate_and_respond(
        prior_thread=starter_thread,
        student_email=student_email,
    )

    # ------------------------------------------------------------
    # 5. Display the grading to the console
    # ------------------------------------------------------------
    print("=== GRADING RESULTS ===\n")
    print(f"Scenario: {result.grading.scenario_name}")
    print(f"Total Score: {result.grading.total_score}/{result.grading.max_total_score}\n")

    print("Rubric Breakdown:")
    for score in result.grading.scores:
        print(f"  - {score.name}: {score.score}/{score.max_score}")

    print("\nOverall Comment:\n")
    print(result.grading.overall_comment)

    print("\nSuggested Revision:\n")
    print(result.grading.revision_example)

    print("\n=====================================================\n")

    # ------------------------------------------------------------
    # 6. Show the AI’s reply as the manager/client
    # ------------------------------------------------------------
    print("=== AI COUNTERPART REPLY (Manager) ===\n")
    print(result.counterpart_reply)

    print("\n=====================================================\n")

    # ------------------------------------------------------------
    # 7. Convert the grading result to a JSON dict for Supabase
    # ------------------------------------------------------------
    supabase_payload = grading_result_to_storage(result.grading)

    print("=== JSON STORED IN SUPABASE ===")
    print(json.dumps(supabase_payload, indent=2))


if __name__ == "__main__":
    main()
