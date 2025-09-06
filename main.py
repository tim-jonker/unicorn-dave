import os
import asyncio
from dataclasses import dataclass
from typing import Dict

import streamlit as st
from pydantic import BaseModel, Field
from pydantic_ai import Agent


# ---------------------------
# Domain model & DB wrapper
# ---------------------------
@dataclass
class House:
    address: str
    price: float
    num_bedrooms: int
    num_bathrooms: int
    square_feet: int


def _default_db() -> Dict[str, House]:
    return {
        "123 Main St": House(
            address="123 Main St",
            price=350000,
            num_bedrooms=3,
            num_bathrooms=2,
            square_feet=1500,
        ),
        "456 Oak Ave": House(
            address="456 Oak Ave",
            price=450000,
            num_bedrooms=4,
            num_bathrooms=3,
            square_feet=2000,
        ),
        "789 Pine Rd": House(
            address="789 Pine Rd",
            price=250000,
            num_bedrooms=2,
            num_bathrooms=1,
            square_feet=900,
        ),
    }


class DatabaseConn:
    """Async facade around the in-memory DB kept in st.session_state."""

    async def house_price(self, address: str) -> float:
        house = st.session_state.house_db.get(address)
        if house:
            return float(house.price)
        raise ValueError("House not found")


# ---------------------------
# Pydantic output schema
# ---------------------------
class TaskOutput(BaseModel):
    response_text: str = Field(
        description="The response text to the client with the refined task"
    )
    pro_required: bool = Field(description="Whether a pro contractor is required")
    urgency: int = Field(description="Urgency level from 1 to 10")


@dataclass
class FakeOutput:
    output: TaskOutput


# ---------------------------
# Agent definition (pydantic-ai)
# ---------------------------
refiner_agent = Agent[DatabaseConn, TaskOutput](
    model="openai:gpt-4o",
    system_prompt=(
        "You are a home renovation expert. Given a task description, determine if a professional "
        "contractor is needed and the urgency level. Give a brief description of the steps to execute "
        "the task and also dedicate a section to describing the benefit of hiring a professional contractor "
        "for this task. If the user gives an address, consult the tool get_house_price to retrieve the house price."
    ),
)


@refiner_agent.tool
async def get_house_price(ctx, address: str) -> float:
    """Look up the price of a house by full address (exact match)."""
    return await ctx.deps.house_price(address)


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Renovation Task Refiner", page_icon="🏠", layout="wide")

# Ensure API key is set (supports either env var or st.secrets)
openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

# Initialize DB in session_state
if "house_db" not in st.session_state:
    st.session_state.house_db: Dict[str, House] = _default_db()

st.title("🏠 Unicorn Dave")

with st.sidebar:
    st.image("unicorn.png", use_container_width=True)
    st.header("Add / Update House")
    with st.form("house_form", clear_on_submit=False):
        addr = st.text_input("Address", placeholder="e.g. 100 Elm St")
        col1, col2 = st.columns(2)
        with col1:
            price = st.number_input(
                "Price ($)", min_value=0.0, step=1000.0, value=350000.0
            )
            beds = st.number_input("Bedrooms", min_value=0, step=1, value=3)
        with col2:
            baths = st.number_input("Bathrooms", min_value=0, step=1, value=2)
            sqft = st.number_input("Square feet", min_value=0, step=50, value=1200)
        submitted = st.form_submit_button("Save house")

    if submitted:
        if not addr.strip():
            st.error("Please provide an address before saving.")
        else:
            st.session_state.house_db[addr.strip()] = House(
                address=addr.strip(),
                price=float(price),
                num_bedrooms=int(beds),
                num_bathrooms=int(baths),
                square_feet=int(sqft),
            )
            st.success(f"Saved: {addr}")

    st.divider()
    # st.caption("Database preview")
    # if st.session_state.house_db:
    #     preview = [{
    #         "address": h.address,
    #         "price": h.price,
    #         "bedrooms": h.num_bedrooms,
    #         "bathrooms": h.num_bathrooms,
    #         "sqft": h.square_feet,
    #     } for h in st.session_state.house_db.values()]
    #     st.dataframe(preview, use_container_width=True, hide_index=True)
    # else:
    #     st.info("No houses yet. Use the form above to add one.")

# Main interaction
left, right = st.columns([2, 1])

with left:
    st.subheader("Describe your task")
    user_prompt = st.text_area(
        "What do you want to do?",
        placeholder="e.g., I want to put up two large paintings, how to do that?",
        height=140,
    )

    addr_options = sorted(st.session_state.house_db.keys())
    st.subheader("House context (optional)")
    if addr_options:
        selected_addr = st.selectbox(
            "Choose an address to provide price context", options=[""] + addr_options
        )
    else:
        selected_addr = ""

    run_clicked = st.button("Run agent ✨", type="primary")

    if run_clicked:
        if not openai_key:
            st.error(
                "Missing OPENAI_API_KEY. Set it as an environment variable or add to st.secrets to continue."
            )
        elif not user_prompt.strip():
            st.error("Please provide a task description.")
        else:
            # Build dependencies: we pass the DB connection; address is passed via the prompt itself.
            deps = DatabaseConn()

            async def _run():
                # return await refiner_agent.run(
                #     deps=deps,
                #     user_prompt=(
                #         user_prompt
                #         if not selected_addr
                #         else f"Address: {selected_addr}. Task: {user_prompt}"
                #     ),
                # )

                return FakeOutput(
                    TaskOutput(
                        response_text="David kan je een API aanvragen en credits erop zetten en die met mij delen?",
                        pro_required=True,
                        urgency=10,
                    )
                )

            with st.spinner("Thinking..."):
                try:
                    result = asyncio.run(_run())
                except RuntimeError as e:
                    # Handle potential 'event loop is running' issues by creating a new loop
                    loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(_run())
                    finally:
                        loop.close()
                except Exception as e:
                    st.exception(e)
                else:
                    st.success("Done!")
                    st.write("### Result")
                    st.markdown(result.output.response_text)
                    st.write("### Assessment")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric(
                            "Pro contractor required",
                            "Yes" if result.output.pro_required else "No",
                        )
                    with c2:
                        st.metric("Urgency (1-10)", int(result.output.urgency))

with right:
    st.subheader("Tips")
    st.markdown(
        """
        - Use the **sidebar** to add or update houses. The agent can call a tool to fetch the price by address.
        - In the main area, type your renovation task and optionally select an address to give the model more context.
        - Results include the reasoning, whether a **pro** is recommended, and an **urgency** score.
        """
    )

st.caption("Powered by pydantic-ai · Streamlit")
