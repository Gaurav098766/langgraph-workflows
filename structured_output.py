from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
load_dotenv()


class ProductReview(BaseModel):
    product_name: str = Field(..., description="The name of the product being reviewed.")
    sentiment: str = Field(..., description="The overall sentiment of the review, e.g., 'positive', 'negative', or 'neutral'.")
    rating: int = Field(..., description="The rating given to the product, on a scale from 1 to 5.", ge=1, le=5)
    pros: List[str] = Field(..., description="A list of pros for the product.")
    cons: List[str] = Field(..., description="A list of cons for the product.")
    summary: str = Field(..., description="Brief summary of the review.")



llm = ChatOpenAI(model="openai/gpt-oss-120b:free",base_url="https://openrouter.ai/api/v1")
llm_structured = llm.with_structured_output(schema=ProductReview)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes product reviews."),
    ("human", "{review_text}")
])


response = prompt | llm_structured

review_text = "I recently bought the XYZ headphones and I'm quite impressed. The sound quality is excellent, and they are very comfortable to wear for long periods. However, the battery life is a bit disappointing, lasting only about 4 hours on a full charge. Overall, I would give it a 4 out of 5."

result = response.invoke({"review_text": review_text})

print(result.model_dump_json(indent=2))