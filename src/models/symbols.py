from sqlmodel import SQLModel, Field




class Symbol(SQLModel,table=True):
    __tablename__ = "symbols"
    id: int = Field(default=None, primary_key=True)
    name: str = Field(index=True)

    def __repr__(self): 
        return f"<Symbol id={self.id} name={self.name}>"