from typing import TypedDict, Annotated

class Test(TypedDict):
    number: Annotated[int, "]
    name: Annotated[str, "name"]

test = Test(number=1, name="test", memo = "111")

print(test)
print(type(test["number"]))
print(type(test["name"]))

