from pydantic import BaseModel, Field


class TrainResponse(BaseModel):
    info: str = Field(
        title="Train info", description="Response string from train embedding or hypernetwork task.")
    train_id: str = Field(
        title="Train ID", description="Train ID for your task")


class TrainLogResponse(BaseModel):
    info: str = Field(
        title="Train info", description="Response string from train embedding or hypernetwork task.")
    train_id: str = Field(
        title="Train ID", description="Train ID for your task")


class TrainInfoResponse(BaseModel):
    info: str = Field(
        title="Train info", description="Response string from train embedding or hypernetwork task.")
