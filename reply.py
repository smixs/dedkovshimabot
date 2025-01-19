from __future__ import annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
from openai import AsyncOpenAI
import os

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from supabase import Client
from typing import List

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

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
   [Дата]: "точная цитата из сообщения"
   ```
2. Если нашел несколько упоминаний - приведи все
3. Указывай контекст до и после цитаты
4. Если информации нет - честно скажи об этом

Отвечай на русском языке, кратко и по делу.
"""

chat_assistant = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@chat_assistant.tool
async def search_chat_history(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Ищет релевантные сообщения в истории чата используя гибридный поиск (BM25 + векторный).
    """
    try:
        # Создаем несколько вариантов запроса
        query_variants = [
            user_query,  # Оригинальный запрос
            f"игрушки коллекции {user_query}",  # Контекст игрушек
            f"акции магазины {user_query}",  # Контекст магазинов
            f"собирать коллекционировать {user_query}"  # Контекст коллекционирования
        ]
        
        all_results = []
        seen_dates = set()
        
        # Ищем по каждому варианту запроса
        for query in query_variants:
            query_embedding = await get_embedding(query, ctx.deps.openai_client)
            
            # Гибридный поиск через SQL
            hybrid_search = """
            WITH query_vars AS (
                SELECT 
                    plainto_tsquery('russian', $1) AS ts_query,
                    $2::vector AS q_emb
            ),
            hybrid AS (
                SELECT
                    cd.*,
                    ts_rank_cd(to_tsvector('russian', cd.raw_content), qv.ts_query) AS bm25_score,
                    1 / (1 + (cd.embedding <-> qv.q_emb)) AS vec_score
                FROM chat_days cd, query_vars qv
                WHERE to_tsvector('russian', cd.raw_content) @@ qv.ts_query
                    OR cd.embedding <-> qv.q_emb < 0.8
            )
            SELECT *, (bm25_score + vec_score) / 2 AS similarity
            FROM hybrid
            ORDER BY similarity DESC
            LIMIT $3;
            """
            
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
        print(f"Error searching chat history: {e}")
        return f"Ошибка при поиске: {str(e)}"