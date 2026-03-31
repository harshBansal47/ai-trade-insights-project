from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate , SystemMessagePromptTemplate


# =========================
# SYSTEM PROMPT
# =========================

CRYPTO_SYSTEM_PROMPT = """
You are a production-grade crypto market analysis engine.

You receive structured market data and return a strict JSON trading decision.

Follow all rules strictly.

<INSERT FULL SYSTEM PROMPT HERE>
"""

system_message = SystemMessagePromptTemplate.from_template(
    CRYPTO_SYSTEM_PROMPT
)



# =========================
# HUMAN INPUT TEMPLATE
# =========================

HUMAN_INPUT_TEMPLATE = """
Here is the market data:

{input_data}

Return only valid JSON.
"""

human_message = HumanMessagePromptTemplate.from_template(
    HUMAN_INPUT_TEMPLATE
)

# =========================
# FINAL CHAT PROMPT
# =========================

crypto_analysis_prompt = ChatPromptTemplate.from_messages(
    [
        system_message,
        human_message
    ]
)