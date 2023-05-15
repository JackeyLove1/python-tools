import asyncio
async def main():
    print("Hello, ")
    await asyncio.sleep(1.0)
    print("World!")
if __name__ == "__main__":
    asyncio.run(main())