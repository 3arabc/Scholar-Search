# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import traceback
from typing import List, Optional, Dict, Any
import uvicorn
import sys
import os
import traceback

# 添加项目路径
from pipeline_spar import AcademicSearchTree
from search_engine import MultiSearchAgent
from citation_generator import get_citation_generator
from log import logger

app = FastAPI(title="Scholar Paper Search API with Frontend", version="1.0.0")

from datetime import datetime
import uuid
import json
import os


# 请求模型
class SearchRequest(BaseModel):
    queries: List[str]
    sources: Optional[List[str]] = ["openalex"]
    end_date: Optional[str] = ""
    max_workers: Optional[int] = 3
    batch_size: Optional[int] = 10
    google_serper_key: Optional[str] = ""  # 添加Google Serper Key字段
    use_advanced_search: Optional[bool] = True  # 是否使用高级搜索（包含query改写和rerank）
    max_depth: Optional[int] = 1  # 搜索树最大深度
    relevance_doc_num: Optional[int] = 10  # 相关文档数量
    similarity_threshold: Optional[float] = 0.5  # 相似度阈值
    # 新增筛选字段
    filter_year_start: Optional[int] = None
    filter_year_end: Optional[int] = None
    filter_min_citations: Optional[int] = None
    filter_fields: Optional[List[str]] = []
    sort_by: Optional[str] = 'year'  # 可选值: 'year', 'citations', 'similarity'

# 响应模型
class SearchResponse(BaseModel):
    status: str
    total_papers: int
    query_results: Dict[str, List[Dict[str, Any]]]
    all_papers: Dict[str, Dict[str, Any]]
    query_source_map: Dict[str, str]
    search_tree: Optional[Dict[str, Any]] = None  # 搜索树结构（高级搜索模式）
    valid_papers: Optional[int] = 0  # 新增

# 初始化搜索引擎
multi_search_agent = MultiSearchAgent()

# 历史记录存储文件
HISTORY_FILE = "./search_history.json"
class SearchHistoryEntry:
    def __init__(self, request: SearchRequest, response: SearchResponse):
        self.id = str(uuid.uuid4())[:8]  # 短ID
        self.timestamp = datetime.now().isoformat()
        self.query = request.queries[0] if request.queries else ""  # 取第一个查询作为主标题
        self.request = request.dict()
        self.response = response.dict()

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_history(history_list):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history_list, f, ensure_ascii=False, indent=2)

# 收藏夹存储文件
FAVORITES_FILE = "./favorites.json"

def load_favorites():
    if os.path.exists(FAVORITES_FILE):
        with open(FAVORITES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_favorites(favorites):
    with open(FAVORITES_FILE, 'w', encoding='utf-8') as f:
        json.dump(favorites, f, ensure_ascii=False, indent=2)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回前端页面"""
    html_file = "./index.html"
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h1>Frontend file not found</h1>
                    <p>Please make sure index.html exists in the templates folder</p>
                </body>
            </html>
            """,
            status_code=404
        )

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "message": "Scholar Paper Search API is running"}

@app.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """
    搜索学术论文API

    Args:
        request: 搜索请求参数

    Returns:
        SearchResponse: 搜索结果
    """
    try:
        logger.info(f"Received search request: {request}")

        # 验证输入
        if not request.queries:
            raise HTTPException(status_code=400, detail="Queries list cannot be empty")

        # 临时设置Google Serper Key环境变量
        if request.google_serper_key:
            os.environ["GOOGLE_SERPER_KEY"] = request.google_serper_key
            logger.info(f"Google Serper Key set from request: {request.google_serper_key}")

        filter_config = {
            'year_start': request.filter_year_start,
            'year_end': request.filter_year_end,
            'min_citations': request.filter_min_citations,
            'fields': request.filter_fields,
            'missing_field_pass': True,
        }

        if request.use_advanced_search:
            # 使用高级搜索（包含query改写、意图判断、rerank等完整pipeline）
            response = await _advanced_search(request, filter_config=filter_config, sort_by=request.sort_by)
        else:
            # 使用简单搜索
            response = await _simple_search(request)

        # 保存历史记录（仅当搜索成功且有结果）
        if response.status == "success":
            history = load_history()
            entry = {
                "id": str(uuid.uuid4())[:8],
                "timestamp": datetime.now().isoformat(),
                "query": request.queries[0] if request.queries else "",
                "request": request.dict(),
                "response": response.dict()
            }
            history.insert(0, entry)
            if len(history) > 50:
                history = history[:50]
            save_history(history)

        return response


    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/history")
async def get_history():
    """获取历史记录列表（摘要）"""
    history = load_history()
    # 只返回id, query, timestamp，不返回完整数据
    summaries = [{"id": h["id"], "query": h["query"], "timestamp": h["timestamp"]} for h in history]
    return {"history": summaries}

@app.get("/history/{history_id}")
async def get_history_detail(history_id: str):
    """获取指定历史记录的完整数据"""
    history = load_history()
    for entry in history:
        if entry["id"] == history_id:
            return entry
    raise HTTPException(status_code=404, detail="History entry not found")

@app.get("/favorites")
async def get_favorites():
    """获取所有收藏的论文"""
    return {"favorites": load_favorites()}

@app.post("/favorites/{paper_id}")
async def add_favorite(paper_id: str, paper_data: Dict[str, Any]):
    """添加论文到收藏夹（需要传入完整的论文数据）"""
    favorites = load_favorites()
    # 检查是否已存在
    existing = next((item for item in favorites if item.get("paper_id") == paper_id), None)
    if existing:
        return {"status": "already_exists", "favorites": favorites}
    # 添加新论文（可以只存储必要字段，但为了简单存储完整数据）
    favorites.append(paper_data)
    save_favorites(favorites)
    return {"status": "added", "favorites": favorites}

@app.delete("/favorites/{paper_id}")
async def remove_favorite(paper_id: str):
    """从收藏夹移除论文"""
    favorites = load_favorites()
    favorites = [item for item in favorites if item.get("paper_id") != paper_id]
    save_favorites(favorites)
    return {"status": "removed", "favorites": favorites}

def standardize_paper_data(paper_data, paper_id=None, source='unknown'):
    """
    标准化论文数据格式

    Args:
        paper_data: 原始论文数据（字典或对象）
        paper_id: 论文ID（可选，如果paper_data中没有）
        source: 数据来源标识

    Returns:
        dict: 标准化后的论文数据
    """
    try:
        # 确保paper_data是字典格式
        if not isinstance(paper_data, dict):
            if hasattr(paper_data, '__dict__'):
                paper_data = paper_data.__dict__
            else:
                logger.warning(f"Cannot convert paper_data to dict: {type(paper_data)}")
                # 创建基本字典结构
                paper_data = {
                    'title': str(paper_data) if paper_data else 'Unknown',
                    'abstract': '',
                    'paper_id': paper_id or 'unknown'
                }

        # 标准化论文数据格式
        standardized_paper = {
            'paper_id': paper_data.get('paper_id') or paper_data.get('arxivId') or paper_data.get('id') or paper_id or 'unknown',
            'title': paper_data.get('title', 'No title available'),
            'authors': paper_data.get('authors', []),
            'abstract': paper_data.get('abstract', ''),
            'year': paper_data.get('publicationYear') or paper_data.get('year', ''),
            'publicationYear': paper_data.get('publicationYear') or paper_data.get('year', ''),
            'url': paper_data.get('url', ''),
            'doi': paper_data.get('doi', ''),
            'citationCount': paper_data.get('citationCount', 0),
            'arxivId': paper_data.get('arxivId', ''),
            'arxiv_url': paper_data.get('arxiv_url', paper_data.get('arxivUrl', '')),
            'openalex_id': paper_data.get('openalex_id', ''),
            'fieldsOfStudy': paper_data.get('fieldsOfStudy', ''),
            'referenceCount': paper_data.get('referenceCount', 0),
            'isOpen': paper_data.get('isOpen', False),
            'source': paper_data.get('source', source),
            'sim_score': paper_data.get('sim_score', 0.0),
            'relevance_details': paper_data.get('sim_info_details', paper_data.get('relevance_details', {}))
        }

        # 处理作者字段 - 确保格式一致
        if standardized_paper['authors']:
            if isinstance(standardized_paper['authors'], list):
                author_names = []
                for author in standardized_paper['authors']:
                    if isinstance(author, dict) and 'name' in author:
                        author_names.append(author['name'])
                    elif isinstance(author, str):
                        author_names.append(author)
                    else:
                        author_names.append(str(author))
                standardized_paper['authors'] = author_names
            elif isinstance(standardized_paper['authors'], str):
                # 如果authors是字符串，尝试分割
                standardized_paper['authors'] = [name.strip() for name in standardized_paper['authors'].split(',') if name.strip()]

        # 处理URL字段 - 提供多种链接选项
        if standardized_paper['openalex_id'] and not standardized_paper['url']:
            standardized_paper['url'] = f"https://openalex.org/{standardized_paper['openalex_id']}"

        if standardized_paper['arxivId'] and not standardized_paper['arxiv_url']:
            standardized_paper['arxiv_url'] = f"https://arxiv.org/abs/{standardized_paper['arxivId']}"

        # 处理其他可能的URL字段
        if paper_data.get('pdf_url'):
            standardized_paper['pdf_url'] = paper_data['pdf_url']
        if paper_data.get('openaccess_url'):
            standardized_paper['openaccess_url'] = paper_data['openaccess_url']
        if paper_data.get('landing_page_url'):
            standardized_paper['landing_page_url'] = paper_data['landing_page_url']

        # 确保数值类型字段的类型正确
        try:
            standardized_paper['citationCount'] = int(standardized_paper['citationCount'] or 0)
        except (ValueError, TypeError):
            standardized_paper['citationCount'] = 0

        try:
            standardized_paper['referenceCount'] = int(standardized_paper['referenceCount'] or 0)
        except (ValueError, TypeError):
            standardized_paper['referenceCount'] = 0

        try:
            standardized_paper['sim_score'] = float(standardized_paper['sim_score'] or 0.0)
        except (ValueError, TypeError):
            standardized_paper['sim_score'] = 0.0

        # 确保year是字符串类型
        if standardized_paper['year']:
            standardized_paper['year'] = str(standardized_paper['year'])
            standardized_paper['publicationYear'] = standardized_paper['year']
        # ========== 🆕 引用格式生成 ==========
        from citation_generator import get_citation_generator
        citation_gen = get_citation_generator()
        if standardized_paper.get('title') and standardized_paper.get('title') != 'No title available':
            try:
                standardized_paper['citations'] = citation_gen.generate_all_citations(standardized_paper)
                standardized_paper['citation_apa'] = standardized_paper['citations'].get('apa', '')
                standardized_paper['citation_mla'] = standardized_paper['citations'].get('mla', '')
                standardized_paper['citation_chicago'] = standardized_paper['citations'].get('chicago', '')
                standardized_paper['citation_bibtex'] = standardized_paper['citations'].get('bibtex', '')
                standardized_paper['citation_gb7714'] = standardized_paper['citations'].get('gb7714', '')
            except Exception as e:
                logger.error(f"Failed to generate citations for {paper_id}: {str(e)}")
                standardized_paper['citations'] = {}
                standardized_paper['citation_apa'] = ''
                standardized_paper['citation_mla'] = ''
                standardized_paper['citation_chicago'] = ''
                standardized_paper['citation_bibtex'] = ''
                standardized_paper['citation_gb7714'] = ''
        else:
            standardized_paper['citations'] = {}
            standardized_paper['citation_apa'] = ''
            standardized_paper['citation_mla'] = ''
            standardized_paper['citation_chicago'] = ''
            standardized_paper['citation_bibtex'] = ''
            standardized_paper['citation_gb7714'] = ''

        # 假设标准化后的变量为 standardized_paper  #wsl-73错觉处理
        # 检查关键字段是否存在且非空
        title = standardized_paper.get('title', '').strip()
        abstract = standardized_paper.get('abstract', '').strip()
        paper_id = standardized_paper.get('paper_id', '').strip()
        # 判断是否为有效论文
        is_valid = bool(title and abstract and paper_id)
        standardized_paper['is_valid'] = is_valid

        # 可以额外添加 invalid_reason 字段，用于调试
        if not is_valid:
            missing = []
            if not title: missing.append('title')
            if not abstract: missing.append('abstract')
            if not paper_id: missing.append('paper_id')
            standardized_paper['invalid_reason'] = f"Missing: {', '.join(missing)}"
        else:
            standardized_paper['invalid_reason'] = ''

        return standardized_paper
    except Exception as e:
        logger.error(f"Error standardizing paper data: {str(e)}")
        logger.error(f"Paper data: {paper_data}")
        # 返回错误占位符
        return {
            'paper_id': str(paper_id) if paper_id else 'error',
            'title': f'Error processing paper: {paper_id or "unknown"}',
            'abstract': f'Error: {str(e)}',
            'authors': [],
            'year': '',
            'publicationYear': '',
            'url': '',
            'doi': '',
            'citationCount': 0,
            'arxivId': '',
            'arxiv_url': '',
            'openalex_id': '',
            'fieldsOfStudy': '',
            'referenceCount': 0,
            'isOpen': False,
            'source': 'error',
            'sim_score': 0.0,
            'relevance_details': {}
        }


def process_paper_collection(papers_data, source='unknown', is_dict_format=True, filter_invalid=True): #wsl-73错觉
    """
    批量处理论文数据集合

    Args:
        papers_data: 论文数据集合（字典或列表）
        source: 数据来源标识
        is_dict_format: 是否为字典格式（True: {paper_id: paper_data}, False: [paper_data, ...])

    Returns:
        tuple: (papers_list, papers_dict) - 论文列表和论文字典
    """
    papers_list = []
    papers_dict = {}

    try:
        if is_dict_format and isinstance(papers_data, dict):
            # 处理字典格式 {paper_id: doc_info}
            for paper_id, paper_info in papers_data.items():
                try:
                    standardized_paper = standardize_paper_data(paper_info, paper_id, source)
                    papers_list.append(standardized_paper)
                    papers_dict[paper_id] = standardized_paper
                except Exception as paper_error:
                    logger.error(f"Error processing paper {paper_id}: {str(paper_error)}")
                    # 创建错误占位符
                    error_paper = {
                        'paper_id': str(paper_id),
                        'title': f'Error processing paper: {paper_id}',
                        'abstract': f'Error: {str(paper_error)}',
                        'authors': [],
                        'year': '',
                        'url': '',
                        'citationCount': 0,
                        'source': 'error'
                    }
                    papers_list.append(error_paper)
                    papers_dict[str(paper_id)] = error_paper

        elif isinstance(papers_data, list):
            # 处理列表格式 [paper_data, ...]
            for i, paper_info in enumerate(papers_data):
                try:
                    # 尝试获取paper_id，如果没有则生成一个
                    if isinstance(paper_info, dict):
                        paper_id = paper_info.get('paper_id', paper_info.get('arxivId', paper_info.get('id', f'paper_{i}')))
                    else:
                        paper_id = f'paper_{i}'

                    standardized_paper = standardize_paper_data(paper_info, paper_id, source)
                    papers_list.append(standardized_paper)
                    papers_dict[paper_id] = standardized_paper
                except Exception as paper_error:
                    logger.error(f"Error processing paper {i}: {str(paper_error)}")
                    continue
        else:
            logger.warning(f"Unexpected papers_data format: {type(papers_data)}")

    except Exception as e:
        logger.error(f"Error processing paper collection: {str(e)}")

    if filter_invalid: #wsl-73错觉
        # 过滤掉无效论文
        papers_list = [p for p in papers_list if p.get('is_valid', False)]
        # 更新字典
        papers_dict = {p['paper_id']: p for p in papers_list}

    return papers_list, papers_dict


async def _advanced_search(request: SearchRequest, filter_config: dict = None, sort_by: str = 'year') -> SearchResponse:
    """
    高级搜索模式，使用AcademicSearchTree进行完整的搜索流程
    包含query改写、意图判断、rerank等功能
    """
    try:
        # 为每个查询创建搜索树
        all_results = {}
        all_papers = {}
        query_source_map = {}
        search_trees = {}

        for query in request.queries:
            logger.info(f"Processing query with advanced search: {query}")

            try:
                # 创建学术搜索树实例
                search_agent = AcademicSearchTree(
                    max_depth=request.max_depth,
                    max_docs=request.relevance_doc_num,
                    similarity_threshold=request.similarity_threshold
                )

                # 执行搜索（包含完整pipeline）
                sorted_docs = search_agent.search(
                    query,
                    end_date=request.end_date,
                    filter_params=filter_config,
                    sort_by=sort_by
                )

                if not sorted_docs:
                    logger.warning(f"No documents found for query: {query}")
                    all_results[query] = []
                    query_source_map[query] = "advanced_search"
                    continue

                logger.info(f"Advanced search returned {len(sorted_docs)} documents for query: {query}")

                # 使用统一的数据处理函数
                if isinstance(sorted_docs, dict):
                    papers_list, papers_dict = process_paper_collection(sorted_docs, 'advanced_search', is_dict_format=True, filter_invalid=True)
                elif isinstance(sorted_docs, list):
                    papers_list, papers_dict = process_paper_collection(sorted_docs, 'advanced_search', is_dict_format=False, filter_invalid=True)
                else:
                    logger.warning(f"Unexpected sorted_docs format: {type(sorted_docs)}")
                    papers_list, papers_dict = [], {}

                all_results[query] = papers_list
                all_papers.update(papers_dict)
                query_source_map[query] = "advanced_search"
                # 保存搜索树结构
                try:
                    if hasattr(search_agent, 'root') and search_agent.root:
                        search_trees[query] = search_agent.root.convert_to_dict()
                    else:
                        logger.warning(f"No search tree root found for query: {query}")
                except Exception as tree_error:
                    logger.error(f"Error converting search tree to dict: {str(tree_error)}")

            except Exception as query_error:
                logger.error(f"Error processing query '{query}': {str(query_error)}")
                logger.error(f"Query error traceback: {traceback.format_exc()}")
                # 为失败的查询添加空结果
                all_results[query] = []
                query_source_map[query] = "error"

        # 构造响应
        response = SearchResponse(
            status="success",
            total_papers=len(all_papers),
            query_results=all_results,
            all_papers=all_papers,
            query_source_map=query_source_map,
            search_tree=search_trees,
            valid_papers = sum(1 for p in all_papers.values() if p.get('is_valid', False))  # wsl-73
        )
        '''
        #wsl-76 ========== 🆕 只保留得分最高的 Top 5 ==========
        if all_papers:
            # 1. 按 sim_score 降序排序
            sorted_papers = sorted(
                all_papers.items(),
                #key=lambda x: x[1].get('sim_score', 0),
                key=lambda x: x[1].get('rerank_score', x[1].get('sim_score', 0)),
                reverse=True
            )
            top5_ids = [pid for pid, _ in sorted_papers[:5]]
            # 2. 更新 all_papers
            all_papers = {pid: all_papers[pid] for pid in top5_ids}
            # 3. 更新 query_results (all_results)
            for query in all_results:
                all_results[query] = [
                    p for p in all_results[query]
                    if p.get('paper_id') in top5_ids
                ]
            # 4. 更新 total_papers
            total_papers = len(all_papers)
        else:
            total_papers = 0
        # 构造响应（使用过滤后的数据）
        response = SearchResponse(
            status="success",
            total_papers=total_papers,  # 更新后的数量
            query_results=all_results,  # 过滤后的 query_results
            all_papers=all_papers,  # 过滤后的 all_papers
            query_source_map=query_source_map,
            search_tree=search_trees,
            valid_papers=sum(1 for p in all_papers.values() if p.get('is_valid', False))
        )
        '''
        logger.info(f"Advanced search completed successfully. Found {len(all_papers)} papers")
        return response

    except Exception as e:
        logger.error(f"Advanced search failed: {str(e)}")
        logger.error(f"Advanced search traceback: {traceback.format_exc()}")
        # 返回错误响应
        return SearchResponse(
            status="error",
            total_papers=0,
            query_results={},
            all_papers={},
            query_source_map={},
            search_tree={"error": str(e)},
            valid_papers=sum(1 for p in all_papers.values() if p.get('is_valid', False))  # wsl-73
        )


async def _simple_search(request: SearchRequest, filter_config: dict = None) -> SearchResponse:
    """
    简单搜索模式，使用MultiSearchAgent进行基础搜索
    """
    try:
        # 更新搜索引擎参数
        multi_search_agent.max_workers = request.max_workers
        multi_search_agent.batch_size = request.batch_size

        # 执行搜索
        query_results, all_papers, query_source_map, query_keywords2raw = multi_search_agent.search_papers(
            querys=request.queries,
            sources=request.sources,
            end_date=request.end_date,
            searched_docs={},
            rerank=True
        )

        # 标准化处理结果数据，确保与前端期望的格式一致
        standardized_query_results = {}
        standardized_all_papers = {}

        # 处理query_results - 每个query对应一个论文列表
        for query, papers in query_results.items():
            papers_list, _ = process_paper_collection(papers, 'simple_search', is_dict_format=False, filter_invalid=True)
            standardized_query_results[query] = papers_list

        # 处理all_papers - 字典格式 {paper_id: paper_data}
        _, standardized_all_papers = process_paper_collection(all_papers, 'simple_search', is_dict_format=True, filter_invalid=True)

        # 构造响应
        response = SearchResponse(
            status="success",
            total_papers=len(standardized_all_papers),
            query_results=standardized_query_results,
            all_papers=standardized_all_papers,
            query_source_map=query_source_map,
            search_tree=None,  # 简单搜索不生成搜索树
            valid_papers=sum(1 for p in all_papers.values() if p.get('is_valid', False))  # wsl-73
        )

        logger.info(f"Simple search completed successfully. Found {len(standardized_all_papers)} papers")
        return response

    except Exception as e:
        logger.error(f"Simple search failed: {str(e)}")
        logger.error(f"Simple search traceback: {traceback.format_exc()}")
        # 返回错误响应而不是重新抛出异常
        return SearchResponse(
            status="error",
            total_papers=0,
            query_results={},
            all_papers={},
            query_source_map={},
            search_tree={"error": str(e)},
            valid_papers=sum(1 for p in all_papers.values() if p.get('is_valid', False))  # wsl-73
        )


@app.get("/sources")
async def get_available_sources():
    """获取可用的搜索源"""
    return {
        "available_sources": ["arxiv", "openalex", "pubmed"],  # 移除semantic scholar
        "description": {
            "arxiv": "ArXiv papers via Google Scholar",
            "openalex": "OpenAlex database",
            "pubmed": "PubMed medical papers"
        }
    }

@app.get("/search-modes")
async def get_search_modes():
    """获取搜索模式信息"""
    return {
        "modes": {
            "simple": {
                "name": "Simple Search",
                "description": "Basic multi-source search with reranking",
                "features": ["Multi-source search", "Basic reranking", "Fast results"]
            },
            "advanced": {
                "name": "Advanced Search",
                "description": "Complete pipeline with query rewriting, intent analysis, and advanced reranking",
                "features": [
                    "Query rewriting and expansion",
                    "Intent analysis and classification",
                    "Reference-based search",
                    "Advanced reranking algorithms",
                    "Search tree visualization",
                    "Iterative refinement"
                ]
            }
        }
    }


@app.get("/paper/{paper_id}/citation")
async def get_paper_citation(paper_id: str, style: str = "apa"):
    """
    获取单篇论文的引用格式

    Args:
        paper_id: 论文ID
        style: 引用格式 (apa, mla, chicago, bibtex, gb7714)
    """
    # 从数据库或搜索结果中获取论文信息
    from local_db_v2 import ArxivDatabase, db_path
    try:
        with ArxivDatabase(db_path) as db:
            paper = db.get(paper_id)  # 从数据库查询
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    citation_gen = get_citation_generator()
    style_map = {
        "apa": citation_gen.format_apa,
        "mla": citation_gen.format_mla,
        "chicago": citation_gen.format_chicago,
        "bibtex": citation_gen.generate_bibtex,
        "gb7714": citation_gen.format_gb7714,
        "all": citation_gen.generate_all_citations
    }

    if style not in style_map:
        raise HTTPException(status_code=400, detail=f"Unsupported style: {style}")

    if style == "all":
        result = style_map[style](paper)
    else:
        result = style_map[style](paper)

    return {
        "paper_id": paper_id,
        "title": paper.get("title", ""),
        "style": style,
        "citation": result
    }

if __name__ == "__main__":
    # 创建templates目录
    templates_dir = "./api/templates"
    os.makedirs(templates_dir, exist_ok=True)

    uvicorn.run(
        "demo_app_with_front:app",
        host="127.0.0.1",  # "0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )