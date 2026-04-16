from pymongo import MongoClient, errors

uri = "mongodb+srv://test:Test123@insoz.elylyhc.mongodb.net/"

try:
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=5000,  # таймаут ожидания сервера
        connectTimeoutMS=5000           # таймаут при установке соединения
    )
    print("Попытка пинга…")
    client.admin.command("ping")
    print("Успешно подключён!")
except errors.ServerSelectionTimeoutError as e:
    print("ServerSelectionTimeoutError:", e)
except Exception as e:
    print("Другая ошибка:", e)