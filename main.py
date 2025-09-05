import asyncio
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent


@dataclass
class House:
    address: str
    price: float
    num_bedrooms: int
    num_bathrooms: int
    square_feet: int

HOUSE_DB = {
    "123 Main St": House(address="123 Main St", price=350000, num_bedrooms=3, num_bathrooms=2, square_feet=1500),
    "456 Oak Ave": House(address="456 Oak Ave", price=450000, num_bedrooms=4, num_bathrooms=3, square_feet=2000),
    "789 Pine Rd": House(address="789 Pine Rd", price=250000, num_bedrooms=2, num_bathrooms=1, square_feet=900),
}

class DatabaseConn:
    async def house_price(self, address: str) -> float:
        house = HOUSE_DB.get(address)
        if house:
            return house.price
        else:
            raise ValueError("House not found")

@dataclass
class TaskDependency:
    address: str
    db: DatabaseConn

class TaskOutput(BaseModel):
    response_text: str = Field(description="The response text to the client with the refined task")
    pro_required: bool = Field(description="Whether a pro contracter is required")
    urgency: int = Field(description="Urgency level from 1 to 10")


refiner_agent = Agent[TaskDependency, TaskOutput](
    model="openai:gpt-4o",
    system_prompt="You are a home renovation expert. Given a task description, determine if a professional contractor is needed and the urgency level. Give a brief description of the steps to execute the task and also dedicate a section to describing the benefit of hiring a professional contractor for this task. Use the database to get house prices.",
)


async def main():
    deps = TaskDependency(address="123 Main St", db=DatabaseConn())
    result = await refiner_agent.run(
        deps=deps,
        user_prompt="I want to put up two large paintings, how to do that?",
    )

    print(result.output)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())
