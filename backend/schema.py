from pydantic import BaseModel, Field


class ChurnInput(BaseModel):
    Tenure: int = Field(..., alias="Tenure")
    Usage_Frequency: int = Field(..., alias="Usage Frequency")
    Support_Calls: int = Field(..., alias="Support Calls")
    Payment_Delay: int = Field(..., alias="Payment Delay")
    Subscription_Type: str = Field(..., alias="Subscription Type")
    Contract_Length: str = Field(..., alias="Contract Length")
    Total_Spend: float = Field(..., alias="Total Spend")
    Last_Interaction: int = Field(..., alias="Last Interaction")
