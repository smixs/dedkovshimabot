import os
from typing import Optional
import logging
from aiogram import Bot, Dispatcher, types
from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize clients
supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", "")
)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN", ""))
dp = Dispatcher(bot)

async def get_embedding(text: str) -> Optional[list[float]]:
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None

async def search_similar_days(query: str, limit: int = 5) -> list[dict]:
    try:
        embedding = await get_embedding(query)
        if not embedding:
            return []
        
        # Using the hybrid search function from the database
        result = supabase.rpc(
            'hybrid_search',
            {
                'query_text': query,
                'query_embedding': embedding,
                'match_count': limit
            }
        ).execute()
        
        return result.data
    except Exception as e:
        logger.error(f"Error searching similar days: {e}")
        return []

async def generate_response(context: list[dict], query: str) -> str:
    try:
        # Prepare context for the model
        context_text = "\n\n".join([
            f"Date: {day['date']}\n{day['raw_content']}"
            for day in context
        ])
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Using the specified model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided chat history context."},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response at this time."

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply("Hi! I'm your chat history assistant. Ask me anything about past conversations!")

@dp.message_handler()
async def handle_message(message: types.Message):
    try:
        # Search for relevant context
        similar_days = await search_similar_days(message.text)
        
        if not similar_days:
            await message.reply("I couldn't find any relevant information in the chat history.")
            return
        
        # Generate response using the context
        response = await generate_response(similar_days, message.text)
        await message.reply(response)
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await message.reply("Sorry, something went wrong. Please try again later.")

async def main():
    try:
        logger.info("Starting bot...")
        await dp.start_polling()
    finally:
        await bot.session.close()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main()) 