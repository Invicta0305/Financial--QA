"""
PDF Processing Utilities for Financial Document Analysis.

This module handles PDF partitioning, table extraction, and chart analysis
using Unstructured API and Groq Vision models.
"""

import time
import re
import os
import base64
import io
import pandas as pd
from groq import Groq, RateLimitError
from langchain_classic.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor, as_completed

import unstructured_client
from unstructured_client.models import shared, errors, operations

from config import UNSTRUCTURED_API_KEY, GROQ_API_KEY


unstructured_api_client = unstructured_client.UnstructuredClient(
    api_key_auth=UNSTRUCTURED_API_KEY
)
groq_vision_client = Groq(api_key=GROQ_API_KEY)


def analyze_chart_with_groq(image_base64):
    """
    Extract structured data from charts and graphs using Groq Vision API.
    
    Args:
        image_base64: Base64 encoded image string
        
    Returns:
        str: Extracted text content or None if extraction fails
    """
    if not image_base64:
        return None
    
    # ~5MB limit on image size
    if len(image_base64) > 5000000:
        print("  Warning: Image exceeds size limit, skipping vision analysis")
        return None
    
    try:
        print(f"  Analyzing chart with Groq Vision API...")
        
        response = groq_vision_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract ALL data from this financial chart/graph/table. Include:
- Chart/table title
- All numeric values with their labels
- Axis labels and units
- Column/row headers
- Trends or patterns
- Any annotations or notes

Format as clear, structured text."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_completion_tokens=2048
        )
        
        extracted_text = response.choices[0].message.content
        print(f"  Successfully extracted {len(extracted_text)} characters")
        return extracted_text
    
    except Exception as e:
        print(f"  Vision extraction failed: {str(e)[:100]}")
        return None


def generate_summary_with_llm(content, llm, content_type):
    """
    Generate concise summary of table or text content using LLM.
    
    Args:
        content: The content to summarize
        llm: Language model instance
        content_type: Type of content (e.g., 'table', 'text')
        
    Returns:
        str: Generated summary
    """
    prompt = f"""You are a financial analyst. Summarize the following {content_type} for retrieval.
Be concise but preserve all important numerical and categorical information.

{content_type}:
{content}

Summary:"""
    
    # Retries up to MAX_RETRIES times on rate limit, then falls back to raw
    # content instead of looping forever.
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            response = llm.invoke(prompt)
            return response.content
        except RateLimitError as e:
            wait_time = 60
            try:
                match = re.search(r'(\d+\.?\d*)s', str(e))
                if match:
                    wait_time = min(float(match.group(1)), 90)
            except Exception:
                pass
            if attempt < MAX_RETRIES - 1:
                print(f"Rate limit. Retrying in {wait_time:.0f}s (attempt {attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait_time)
            else:
                print(f"Rate limit: max retries hit, using raw content fallback.")
                return content[:500] + "..." if len(content) > 500 else content
        except Exception as e:
            print(f"Summary generation failed: {e}")
            return content[:500] + "..." if len(content) > 500 else content
    return content[:500] + "..." if len(content) > 500 else content


def api_partition_pdf(pdf_path):
    """
    Partition PDF using Unstructured Cloud API with hi_res strategy.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        list: List of extracted document elements
    """
    with open(pdf_path, "rb") as f:
        data = f.read()

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=data, 
                file_name=os.path.basename(pdf_path)
            ),
            strategy=shared.Strategy.HI_RES,
            languages=["eng"],
            extract_image_block_types=["Image", "Table"],
        )
    )
    
    try:
        res = unstructured_api_client.general.partition(request=req)
        print("PDF partitioning completed successfully")
    except errors.UnstructuredClientError as e:
        print(f"Error during PDF partitioning: {e.message}")
        return []

    # Wrap each raw API response dict in a simple object so downstream code
    # can use attribute access (el.category, el.metadata.page_number, etc.)
    class Element:
        def __init__(self, el):
            self.category = el.get("type") or el.get("category")
            self.text = el.get("text", "")
            
            metadata_dict = el.get("metadata", {})
            self.metadata = type("Metadata", (), {
                "page_number": metadata_dict.get("page_number", 0),
                "text_as_html": metadata_dict.get("text_as_html", ""),
                "image_base64": metadata_dict.get("image_base64", ""),
                "image_path": metadata_dict.get("image_path", ""),
            })()

    elements = [Element(el) for el in res.elements]
    
    visual_count = sum(1 for el in elements if el.category in ["Image", "Figure"])
    table_count = sum(1 for el in elements if el.category == "Table")
    print(f"Extracted {len(elements)} elements: {visual_count} images, {table_count} tables")
    
    return elements


def local_partition_pdf(pdf_path):
    """
    Fallback method for local PDF processing without API.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        list: List of extracted document elements
    """
    from unstructured.partition.pdf import partition_pdf
    
    return partition_pdf(
        filename=pdf_path, 
        strategy="hi_res",
        infer_table_structure=True
    )


def partition_pdf_flexible(pdf_path):
    """
    Partition PDF using cloud API if available, otherwise fall back to local processing.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        list: List of extracted document elements
    """
    if UNSTRUCTURED_API_KEY:
        print("Using Unstructured Cloud API with Groq Vision integration")
        return api_partition_pdf(pdf_path)
    else:
        print("Using local Unstructured processing")
        return local_partition_pdf(pdf_path)


def process_single_element(element, llm, pdf_path, min_summary_length=300):
    """
    Process a single document element (table, image, or text).
    
    Args:
        element: Document element to process
        llm: Language model for summarization
        pdf_path: Source PDF file path
        min_summary_length: Minimum length to trigger summarization
        
    Returns:
        list: List of processed Document objects
    """
    docs = []
    meta = {
        "source": os.path.basename(pdf_path),
        "page": getattr(element.metadata, 'page_number', 0),
        "type": element.category
    }
    
    try:
        if element.category == "Table":
            html_content = getattr(element.metadata, 'text_as_html', "")
            
            if html_content:
                try:
                    df = pd.read_html(io.StringIO(html_content))[0]
                    markdown_table = df.to_markdown(index=False)
                    
                    docs.append(Document(
                        page_content=f"TABLE:\n{markdown_table}", 
                        metadata=meta
                    ))
                    
                    if len(markdown_table) > min_summary_length:
                        summary = generate_summary_with_llm(markdown_table, llm, "table")
                        docs.append(Document(
                            page_content=f"TABLE_SUMMARY:\n{summary}", 
                            metadata={**meta, "type": "table_summary"}
                        ))
                except Exception as e:
                    print(f"Table parsing error on page {meta['page']}: {e}")
                    text = getattr(element, "text", "")
                    if text:
                        docs.append(Document(
                            page_content=f"TABLE:\n{text}", 
                            metadata=meta
                        ))
            else:
                text = getattr(element, "text", "")
                if text:
                    docs.append(Document(
                        page_content=f"TABLE:\n{text}", 
                        metadata=meta
                    ))
        
        elif element.category in ["Image", "Figure"]:
            text = getattr(element, "text", "")
            image_data = getattr(element.metadata, 'image_base64', "")
            
            if image_data:
                print(f"Processing visual element on page {meta['page']}")
                chart_text = analyze_chart_with_groq(image_data)
                
                if chart_text:
                    content = f"CHART/GRAPH DATA (extracted by Groq Vision):\n{chart_text}"
                    docs.append(Document(
                        page_content=content,
                        metadata={**meta, "has_visual": True, "vision_extracted": True}
                    ))
                elif text and text.strip():
                    docs.append(Document(
                        page_content=f"VISUAL ELEMENT:\n{text}",
                        metadata={**meta, "has_visual": True}
                    ))
                else:
                    docs.append(Document(
                        page_content="[VISUAL ELEMENT: Chart/image detected but content not extracted]",
                        metadata={**meta, "has_visual": True}
                    ))
            elif text and text.strip():
                docs.append(Document(
                    page_content=f"VISUAL ELEMENT:\n{text}",
                    metadata={**meta, "has_visual": True}
                ))
        
        else:
            text = getattr(element, "text", "")
            if text and text.strip():
                docs.append(Document(page_content=text, metadata=meta))
    
    except Exception as e:
        print(f"Error processing {element.category} on page {meta['page']}: {e}")
    
    return docs


def process_elements_parallel(elements, llm, pdf_path, max_workers=1):
    """
    Process document elements in parallel with progress tracking.
    
    Args:
        elements: List of document elements
        llm: Language model instance
        pdf_path: Source PDF file path
        max_workers: Number of parallel workers
        
    Returns:
        list: List of processed Document objects
    """
    print(f"Processing {len(elements)} elements with {max_workers} worker(s)...")
    
    all_docs = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_element = {
            executor.submit(process_single_element, elem, llm, pdf_path): elem
            for elem in elements
        }
        
        for i, future in enumerate(as_completed(future_to_element), 1):
            try:
                docs = future.result()
                all_docs.extend(docs)
                
                if i % 10 == 0:
                    print(f"  Processed {i}/{len(elements)} elements...")
                
                # 0.5s delay avoids hitting Groq's rate limit (429) when an element
                # triggers an LLM call (tables and images do, plain text doesn't)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error in parallel processing: {e}")
    
    all_docs = [doc for doc in all_docs if doc.page_content and doc.page_content.strip()]
    
    visual_docs = sum(1 for doc in all_docs if doc.metadata.get("vision_extracted", False))
    table_docs = sum(1 for doc in all_docs if "TABLE" in doc.page_content[:20])
    
    print(f"Created {len(all_docs)} documents ({visual_docs} vision-analyzed, {table_docs} tables)")
    
    return all_docs


def process_elements(elements, llm, pdf_path):
    """
    Main entry point for processing document elements.
    
    Args:
        elements: List of document elements
        llm: Language model instance
        pdf_path: Source PDF file path
        
    Returns:
        list: List of processed Document objects
    """
    # max_workers=1 (sequential) avoids hammering Groq's free-tier rate limit —
    # running several parallel LLM calls (e.g. 4 workers) triggers repeated 429
    # errors and can stall for 20+ minutes before finishing.
    return process_elements_parallel(elements, llm, pdf_path, max_workers=1)