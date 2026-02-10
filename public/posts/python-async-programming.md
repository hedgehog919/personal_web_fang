## 為什麼需要非同步？

在處理 I/O 密集型任務時，同步程式會浪費大量時間等待。

## asyncio 基礎

Python 3.5 引入了 async/await 語法：

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return "data"

async def main():
    result = await fetch_data()
    print(result)

asyncio.run(main())
```

## 實際應用

在開發 QueryMaster Chatbot 時，需要同時查詢多個資料來源，非同步程式設計大幅提升了效能。

## 注意事項

- 不是所有函式庫都支援非同步
- 需要注意 event loop 的管理
- Debug 比同步程式困難
