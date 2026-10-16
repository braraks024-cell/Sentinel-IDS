import unittest

from database import DatabaseConnection


class RecordingPostgresConnection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, object]] = []

    def execute(self, statement: str, parameters: object = ()) -> "RecordingPostgresConnection":
        self.statements.append((statement.strip(), parameters))
        return self

    def cursor(self) -> "RecordingPostgresConnection":
        return self

    def executemany(
        self, statement: str, parameters: object
    ) -> "RecordingPostgresConnection":
        self.statements.append((statement.strip(), parameters))
        return self


class DatabaseConnectionTests(unittest.TestCase):
    def test_postgres_adapter_converts_placeholders_and_runs_schema_statements(self) -> None:
        raw_connection = RecordingPostgresConnection()
        connection = DatabaseConnection(raw_connection, postgres=True)

        connection.execute("SELECT * FROM users WHERE id = ?", (7,))
        connection.executemany(
            "INSERT INTO predictions (analysis_id) VALUES (?)", [(1,), (2,)]
        )
        connection.executescript("CREATE TABLE first (id INT); CREATE TABLE second (id INT);")

        self.assertEqual(
            raw_connection.statements,
            [
                ("SELECT * FROM users WHERE id = %s", (7,)),
                ("INSERT INTO predictions (analysis_id) VALUES (%s)", [(1,), (2,)]),
                ("CREATE TABLE first (id INT)", ()),
                ("CREATE TABLE second (id INT)", ()),
            ],
        )


if __name__ == "__main__":
    unittest.main()
