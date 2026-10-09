from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def make_engine(url: str) -> Engine:
    # connect_timeout: an unreachable Postgres fails in seconds, not after the OS's TCP
    # timeout (75 s+), e.g. while a candidate waits on the line for the agent.
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5})


def make_sessionmaker(engine: Engine) -> sessionmaker:
    return sessionmaker(engine, expire_on_commit=False)
