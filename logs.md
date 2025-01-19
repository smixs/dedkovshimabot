# Project Progress Log

⚠️ **IMPORTANT MODEL NOTE** ⚠️
===============================
DO NOT CHANGE THE MODEL! We use `gpt-4o-mini` which is a valid model!
===============================

## Done:
1. Created initial project structure with UV
2. Set up environment variables in `.env`:
   - OpenAI API key
   - Supabase credentials
   - LLM model configuration

3. Created Supabase schema:
```sql
create table chat_days (
    id bigint primary key generated always as identity,
    date date not null,                    
    raw_content text not null,             
    title text,                            
    summary text,                          
    metadata jsonb,                        
    embedding vector(1536),                
    created_at timestamp with time zone default timezone('utc'::text, now())
);
```

4. Created new simplified extract.py:
   - Removed all classes (ChatMessage, DayChat)
   - Added regex patterns for metadata extraction:
     * URLs: `http[s]?://...`
     * Attachments: `\[Attached:.*?\]`
     * Participants: `\] ([^:]+):`
   - Created metadata extraction function
   - Simplified day processing to preserve raw content
   - Added proper logging

5. Implemented daily processing:
   - Process each day independently
   - Generate summary only for days with >5 words
   - Store raw content for all days
   - Individual embeddings per day
   - Strict save-after-process logic
   - Cost optimization: using gpt-4o-mini ($0.15/1M input)

6. Optimized summary format:
   - Title: 3-4 ключевых слова через запятую
   - Summary: 5-15 буллитов с новой строки
   - Строгий формат без вступлений/заключений
   - Сохранение в базу как текст с переносами
   - Пропуск саммари для коротких дней
   - Сохранение метаданных для всех дней

7. Improved error handling:
   - Graceful handling of short messages
   - Skip summary on errors, but save content
   - Zero vector fallback for embedding errors
   - Detailed logging of all operations
   - Independent processing per day

8. Optimized database indexes:
   - Added B-tree index for date field (2.9ms execution)
   - Added B-tree index for created_at
   - Added HNSW index for vector search:
     * m = 16, ef_construction = 128
     * ef = 64 for search optimization
     * Distance range 0.61-0.65 for similar days
   - Added GIN index for metadata (2.6ms execution)
   - Verified all indexes with EXPLAIN ANALYZE
   - No data reprocessing needed
   - Confirmed semantic search quality

9. Implemented hybrid search:
   - Added GIN index for full-text search (russian)
   - Created hybrid_search SQL function:
     * Combined BM25 and vector similarity
     * Single query optimization
     * Proper column mapping
     * Normalized scoring (0.5 * bm25 + 0.5 * vector)
   - Integrated with query_variants logic
   - Maintained parallel search capabilities
   - Applied directly in Supabase SQL Editor
   - Fixed return type mismatch
   - Simplified score calculation

## Current Structure:
1. Daily processing:
   - Raw content preservation
   - Conditional summary generation
   - Metadata extraction
   - Embedding generation
   - Immediate storage

2. Data handling:
   - Full text in raw_content
   - Optional title/summary
   - Structured metadata (URLs, files, participants)
   - Full-text embeddings for search
   - Proper error states

3. Error handling:
   - Per-day isolation
   - Continues from last successful save
   - Detailed error logging
   - Fallback strategies

## Next Steps:
1. Test and verify:
   - Run initial test with sample days
   - Verify summary quality
   - Check data consistency
   - Test search capabilities
   - Monitor costs

## Next Phase: Telegram Bot Migration
10. Planned changes:
   - Migrate from Streamlit to Telegram bot interface
   - Keep existing RAG functionality
   - Optimize for Docker + KVM2 deployment

11. New minimal structure:
   ```
   deadcough2/
   ├── bot.py           # Main bot file with RAG
   ├── .env             # Configuration
   ├── Dockerfile       # Minimal image
   └── docker-compose.yml
   ```

12. Implementation plan:
   - Move RAG logic from reply.py to bot.py
   - Add Telegram mention handler
   - Setup Docker deployment
   - Basic health monitoring

13. Technical decisions:
   - Single entrypoint (bot.py)
   - Minimal dependencies:
     * aiogram
     * openai
     * supabase
   - Auto-restart in Docker
   - Simple health checks

## Notes:
- Optimized for cost (~$0.52 for 1800 days)
- Immediate storage after processing
- Full context preservation
- Safe error handling
- Detailed logging
- Flexible summary generation
- No migrations needed (Supabase cloud DB):
  * All DB changes via SQL Editor
  * Direct schema modifications
  * No local migration files
  * Clean repository structure
- Keeping existing files for history
- Simplified architecture (DRY, KISS, YAGNI)
- Focus on stability and maintainability
- Ready for KVM2 deployment

