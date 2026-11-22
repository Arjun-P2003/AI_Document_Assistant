
from modules.llm_generator import LLMGenerator


def main():
    print("\n===== OLLAMA LLM GENERATION TEST =====")

    try:
        # Step 1: Initialize the local AI model
        generator = LLMGenerator(
            model_name="qwen2.5:3b"
        )

        print("LLM Generator initialized!")
        print("Model:", generator.model_name)

        # Step 2: Prepare sample document context
        context = """
[Source 1 | Page 1]
Retrieval Augmented Generation uses document
search to answer questions.
"""

        # Step 3: Prepare a question
        question = (
            "What does Retrieval Augmented Generation "
            "use to answer questions?"
        )

        print("\n===== USER QUESTION =====")
        print(question)

        # Step 4: Generate an answer
        print("\nGenerating AI answer...")

        answer = generator.generate_answer(
            question=question,
            context=context
        )

        # Step 5: Display the generated answer
        print("\n===== AI GENERATED ANSWER =====")
        print(answer)

        # Step 6: Verify the response
        assert isinstance(answer, str), (
            "Answer must be a string."
        )

        assert answer.strip(), (
            "Generated answer is empty."
        )

        print(
            "\nSUCCESS: Ollama LLM generation test completed!"
        )

    except Exception as error:
        print("\nERROR:", error)
        raise


if __name__ == "__main__":
    main()
