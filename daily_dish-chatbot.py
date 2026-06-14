"""
The Daily Dish Chatbot – Interactive Console Loop

Lab: Building AI Agents from Scratch with Python
"""

def chatbot(user_input: str) -> str:
    """
    Very simple rule-based chatbot placeholder.
    You can replace this logic with LLM calls later.
    """
    text = user_input.lower()

    if "menu" in text or "today" in text:
        return "Today’s specials are tomato basil soup, grilled salmon, and chocolate lava cake."
    if "vegan" in text:
        return "We have a vegan buddha bowl with quinoa, roasted veggies, and hummus."
    if "recommend" in text or "suggest" in text:
        return "I recommend the grilled salmon with lemon herb butter and a side salad."
    if "thanks" in text or "thank you" in text:
        return "You’re welcome! Let me know if you have any other questions."

    return "I’m not sure, but you can ask about today’s menu, vegan options, or recommendations."


if __name__ == "__main__":
    print("🍽️ Welcome to The Daily Dish Chatbot!")
    print("Type 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ")

        # Exit conditions for ending the chat
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: 👋 Thanks for visiting The Daily Dish!")
            break

        # Process the query and print the chatbot's response
        print("Chatbot:", chatbot(user_input))
        print()
