from pydantic import BaseModel


class CollectionCreate(BaseModel):
    name: str


class CollectionStats(BaseModel):
    name: str
    vectors_count: int
    points_count: int


class CollectionList(BaseModel):
    collections: list[str]