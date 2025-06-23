#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from langchain_openai import OpenAIEmbeddings
from typing import List
import json

class EncodingFixedEmbeddings(OpenAIEmbeddings):
    api_key: str = "Empty"
    def __init__(self, api_key, base_url, model, show_progress_bar=True, **kwargs):
        super().__init__(
            openai_api_key=api_key,
            base_url=base_url,
            model=model,
            show_progress_bar=show_progress_bar,
            **kwargs
        )
        self.api_key = api_key
    """修正編碼問題的 OpenAI Embeddings 類別
    
    這個類別在呼叫 API 之前會確保文字是正確的 UTF-8 編碼，
    並將所有步驟記錄下來以便偵錯。
    """
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多個文件，並確保正確編碼
        
        Args:
            texts: 要嵌入的文字清單
            
        Returns:
            嵌入向量清單
        """
        print(f"嵌入文件數量: {len(texts)}")
        # 直接使用 OpenAI 套件建立嵌入
        from openai import OpenAI
        
        # 獲取 API 金鑰和基礎 URL
        api_key = self.api_key
        api_base = self.openai_api_base or self.client.base_url
        
        # 建立 OpenAI 客戶端
        direct_client = OpenAI(api_key=api_key, base_url=api_base)
        
        # 直接呼叫 API
        try:
            print(f"使用模型 {self.model} 進行嵌入")
            response = direct_client.embeddings.create(
                input=texts,
                model=self.model
            )
            # 將回應轉換為 LangChain 預期的格式
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"直接呼叫嵌入 API 時發生錯誤: {str(e)}")
            print(f"第一個文本的部分內容: {texts[0][:100] if texts else 'None'}")
            # 如果直接呼叫失敗，嘗試使用清理後的文字
            try:
                print("嘗試清理文字後重新嘗試...")
                from document_processor import DocumentProcessor
                processor = DocumentProcessor()
                cleaned_texts = [processor._clean_text_for_embedding(text) for text in texts]
                response = direct_client.embeddings.create(
                    input=cleaned_texts,
                    model=self.model
                )
                return [data.embedding for data in response.data]
            except Exception as e2:
                print(f"清理後仍然失敗: {str(e2)}")
                # 最後嘗試使用父類別的方法
                return super().embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查詢文字，並確保正確編碼
        
        Args:
            text: 要嵌入的查詢文字
            
        Returns:
            嵌入向量
        """
        print(f"嵌入查詢: {text[:50]}...")
        # 直接使用 OpenAI 套件建立嵌入
        from openai import OpenAI
        
        # 獲取 API 金鑰和基礎 URL
        api_key = self.api_key
        api_base = self.openai_api_base or self.client.base_url
        
        # 建立 OpenAI 客戶端
        direct_client = OpenAI(api_key=api_key, base_url=api_base)
        
        # 直接呼叫 API
        try:
            response = direct_client.embeddings.create(
                input=[text],
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"直接呼叫嵌入 API 時發生錯誤: {str(e)}")
            print(f"查詢文本: {text}")
            # 如果直接呼叫失敗，嘗試使用清理後的文字
            try:
                print("嘗試清理文字後重新嘗試...")
                from document_processor import DocumentProcessor
                processor = DocumentProcessor()
                cleaned_text = processor._clean_text_for_embedding(text)
                response = direct_client.embeddings.create(
                    input=[cleaned_text],
                    model=self.model
                )
                return response.data[0].embedding
            except Exception as e2:
                print(f"清理後仍然失敗: {str(e2)}")
                # 最後嘗試使用父類別的方法
                return super().embed_query(text)
