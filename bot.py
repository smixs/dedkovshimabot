from __future__ import annotations

# Standard library
from dataclasses import dataclass
import logging
import asyncio
import os
import sys
from typing import List

# Aiogram
from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatType, MessageEntityType
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.token import TokenValidationError

# Third party
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from supabase import create_client, Client
from dotenv import load_dotenv
import logfire
from pydantic_ai.tools import RunContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logfire.configure(send_to_logfire='if-token-present')

# Load environment variables
load_dotenv()

# Validate required environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not all([TELEGRAM_TOKEN, OPENAI_KEY, SUPABASE_URL, SUPABASE_KEY]):
    logger.error("Missing required environment variables!")
    sys.exit(1)

# Initialize model
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
Ты - ассистент, который помогает находить информацию в истории чата. У тебя есть доступ к базе данных с историей сообщений, их саммари и метаданными.

Твоя задача - находить релевантную информацию и давать полезные ответы на вопросы пользователя.

Когда отвечаешь:
1. ОБЯЗАТЕЛЬНО цитируй найденные сообщения в формате:
   ```
   [Дата],[Имя]: "точная цитата из сообщения"
   ```
2. Если нашел несколько упоминаний - приведи все
3. Указывай контекст до и после цитаты
4. Если информации нет - честно скажи об этом

Отвечай на русском языке, кратко и по делу.
"""

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = AsyncOpenAI(api_key=OPENAI_KEY)

# Initialize router
router = Router(name=__name__)

chat_assistant = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

@chat_assistant.tool
async def search_chat_history(ctx: RunContext[PydanticAIDeps], query: str) -> str:
    """
    Ищет релевантные сообщения в истории чата используя гибридный поиск (BM25 + векторный).
    """
    try:
        # Создаем несколько вариантов запроса
        query_variants = [
            query,  # Оригинальный запрос
            f"игрушки коллекции {query}",  # Контекст игрушек
            f"акции магазины {query}",  # Контекст магазинов
            f"собирать коллекционировать {query}"  # Контекст коллекционирования
        ]
        
        all_results = []
        seen_dates = set()
        
        # Ищем по каждому варианту запроса
        for query in query_variants:
            # Используем openai_client из контекста
            response = await ctx.deps.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = response.data[0].embedding
            
            # Используем supabase из контекста
            matches = ctx.deps.supabase.rpc(
                'hybrid_search',
                {
                    'text_query': query,
                    'query_embedding': query_embedding,
                    'limit_val': 50
                }
            ).execute()
            
            if matches.data:
                for day in matches.data:
                    if day['date'] not in seen_dates:
                        seen_dates.add(day['date'])
                        all_results.append(day)

        if not all_results:
            return "К сожалению, не удалось найти релевантную информацию по вашему запросу."

        # Сортируем по релевантности
        all_results.sort(key=lambda x: float(x.get('similarity', 0)), reverse=True)
        
        # Форматируем результаты
        formatted_chunks = []
        for day in all_results[:30]:  # Берем топ-30
            date = day['date']
            title = day.get('title', '')
            summary = day.get('summary', '')
            content = day['raw_content']
            similarity = float(day.get('similarity', 0))
            
            # Форматируем с учетом всех полей
            chunk_text = f"""
Дата: {date} (релевантность: {similarity:.2f})
{'Тема: ' + title if title else ''}
{'Краткое содержание: ' + summary if summary else ''}
-------------------------------------------
{content}
"""
            formatted_chunks.append(chunk_text)
            
        return "\n\n===================\n\n".join(formatted_chunks)
        
    except Exception as e:
        logger.error(f"Error searching chat history: {e}")
        return f"Ошибка при поиске: {str(e)}"

@router.message(Command("start", "help"))
async def send_welcome(message: Message):
    await message.reply("Привет! Я помогу найти информацию в истории чата. Спрашивай что угодно!")

# Глобальная переменная для хранения информации о боте
bot_info = None

async def main() -> None:
    global bot_info
    
    # Initialize Bot instance with a default parse mode
    bot = Bot(token=TELEGRAM_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    
    try:
        # Получаем информацию о боте один раз при старте
        bot_info = await bot.get_me()
        
        # Initialize Dispatcher
        dp = Dispatcher()
        
        # Register router
        dp.include_router(router)
        
        # Start polling
        await dp.start_polling(bot)
    except TokenValidationError:
        logger.error("Invalid Telegram token!")
        return
    except Exception as e:
        logger.error(f"Bot stopped due to error: {e}")
        return
    finally:
        await bot.session.close()

@router.message(F.text)
async def handle_message(message: Message, bot: Bot) -> None:
    # Простая проверка типа чата
    if message.chat.type != ChatType.PRIVATE:
        # В группах проверяем упоминание или ответ
        is_mentioned = any(
            entity.type == MessageEntityType.MENTION and 
            message.text[entity.offset:entity.offset + entity.length] == f"@{bot_info.username}"
            for entity in (message.entities or [])
        )
        
        is_reply = message.reply_to_message and message.reply_to_message.from_user.id == bot.id
        
        if not (is_mentioned or is_reply):
            return
    
    try:
        result = await chat_assistant.run(
            message.text, 
            deps=PydanticAIDeps(supabase=supabase, openai_client=openai_client)
        )
        await message.answer(result.data)
    except Exception as e:
        logger.exception(f"Error: {e}")
        await message.answer("Sorry, something went wrong.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot stopped due to error: {e}")
        sys.exit(1) 