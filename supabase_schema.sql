-- Enable pgvector
create extension if not exists vector;

-- Parent table: full Dieu context
create table if not exists legal_parent_chunks (
    id text primary key,
    dieu_id text not null,
    van_ban_id text not null,
    content text not null,
    metadata jsonb,
    created_at timestamptz default now()
);

create index if not exists legal_parent_chunks_dieu_id_idx
    on legal_parent_chunks (dieu_id);

create index if not exists legal_parent_chunks_van_ban_id_idx
    on legal_parent_chunks (van_ban_id);

-- Child table: retrieval chunks with embeddings
create table if not exists legal_child_chunks (
    id text primary key,
    dieu_id text not null,
    parent_id text not null,
    van_ban_id text not null,
    chunk_type text not null,
    content text not null,
    embedding vector(1024),
    metadata jsonb,
    created_at timestamptz default now()
);

create index if not exists legal_child_chunks_dieu_id_idx
    on legal_child_chunks (dieu_id);

create index if not exists legal_child_chunks_parent_id_idx
    on legal_child_chunks (parent_id);

create index if not exists legal_child_chunks_van_ban_id_idx
    on legal_child_chunks (van_ban_id);

-- HNSW index for vector search (cosine)
create index if not exists legal_child_chunks_embedding_hnsw
    on legal_child_chunks using hnsw (embedding vector_cosine_ops);
