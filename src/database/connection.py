"""Database connection and session management"""


import os
from typing import Generator, Optional
from contextlib import contextmanager
import logging
from urllib.parse import urlparse


from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.engine import Engine
import redis
from pymongo import MongoClient
from dotenv import load_dotenv

from .models import Base

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration"""

    def __init__(self):
        """  Init  ."""
        self.postgresql_url = os.getenv("DATABASE_URL", "postgresql://localhost/he_graph")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/he_graph")

        # Connection pool settings
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "40"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))

        # Performance settings
        self.echo_sql = os.getenv("ECHO_SQL", "false").lower() == "true"
        self.slow_query_threshold = float(os.getenv("SLOW_QUERY_THRESHOLD", "1.0"))

class DatabaseManager:
    """Manages database connections and sessions"""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """  Init  ."""
        self.config = config or DatabaseConfig()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._scoped_session: Optional[scoped_session] = None
        self._redis_client: Optional[redis.Redis] = None
        self._mongo_client: Optional[MongoClient] = None

    @property
    def engine(self) -> Engine:
        """Get or create SQLAlchemy engine"""
        if self._engine is None:
            self._engine = create_engine(
                self.config.postgresql_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo_sql,
                pool_pre_ping=True,  # Verify connections before using
                connect_args={
                    "connect_timeout": 10,
                    "application_name": "he_graph_embeddings"
                }
            )

            # Add event listeners
            self._setup_event_listeners(self._engine)

        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get or create session factory"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        return self._session_factory

    @property
    def scoped_session(self) -> scoped_session:
        """Get or create scoped session"""
        if self._scoped_session is None:
            self._scoped_session = scoped_session(self.session_factory)
        return self._scoped_session

    @property
    def redis_client(self) -> redis.Redis:
        """Get or create Redis client"""
        if self._redis_client is None:
            url_parts = urlparse(self.config.redis_url)
            self._redis_client = redis.Redis(
                host=url_parts.hostname or "localhost",
                port=url_parts.port or 6379,
                db=int(url_parts.path.lstrip("/")) if url_parts.path else 0,
                password=url_parts.password,
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 5,  # TCP_KEEPINTVL
                    3: 5,  # TCP_KEEPCNT
                },
                connection_pool=redis.ConnectionPool(
                    max_connections=50,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
            )

            # Test connection
            try:
                self._redis_client.ping()
                logger.info("Redis connection established")
            except redis.ConnectionError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise

        return self._redis_client

    @property
    def mongo_client(self) -> MongoClient:
        """Get or create MongoDB client"""
        if self._mongo_client is None:
            self._mongo_client = MongoClient(
                self.config.mongodb_uri,
                maxPoolSize=50,
                minPoolSize=10,
                maxIdleTimeMS=60000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000,
                retryWrites=True,
                retryReads=True
            )

            # Test connection
            try:
                self._mongo_client.admin.command("ping")
                logger.info("MongoDB connection established")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise

        return self._mongo_client

    @property
    def mongo_db(self) -> None:
        """Get MongoDB database"""
        db_name = urlparse(self.config.mongodb_uri).path.lstrip("/") or "he_graph"
        return self.mongo_client[db_name]

    def create_tables(self) -> None:
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")

    def drop_tables(self) -> None:
        """Drop all database tables"""
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope for database operations"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            logger.error(f"Error in operation: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def redis_pipeline(self) -> None:
        """Get Redis pipeline for batch operations"""
        pipe = self.redis_client.pipeline()
        try:
            yield pipe
            pipe.execute()
        except Exception:
            logger.error(f"Error in operation: {e}")
            pipe.reset()
            raise

    def _setup_event_listeners(self, engine -> None: Engine):
        """Setup SQLAlchemy event listeners"""

        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Configure connection on connect"""
            # Enable foreign keys for SQLite
            if "sqlite" in self.config.postgresql_url:
                dbapi_conn.execute("PRAGMA foreign_keys=ON")

        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log slow queries"""
            conn.info.setdefault("query_start_time", []).append(os.time.time())

        @event.listens_for(engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Check query execution time"""
            total_time = os.time.time() - conn.info["query_start_time"].pop(-1)
            if total_time > self.config.slow_query_threshold:
                logger.warning(f"Slow query ({total_time:.2f}s): {statement[:100]}...")

    def close(self) -> None:
        """Close all database connections"""
        if self._engine:
            self._engine.dispose()
            self._engine = None

        if self._scoped_session:
            self._scoped_session.remove()
            self._scoped_session = None

        if self._redis_client:
            self._redis_client.close()
            self._redis_client = None

        if self._mongo_client:
            self._mongo_client.close()
            self._mongo_client = None

        logger.info("All database connections closed")

    def __enter__(self):
        """  Enter  ."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """  Exit  ."""
        self.close()

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions
def get_session() -> Session:
    """Get a new database session"""
    return db_manager.session_factory()

def get_redis() -> redis.Redis:
    """Get Redis client"""
    return db_manager.redis_client

def get_mongodb():
    """Get MongoDB database"""
    return db_manager.mongo_db

@contextmanager
def transaction() -> Generator[Session, None, None]:
    """Database transaction context manager"""
    with db_manager.session_scope() as session:
        yield session