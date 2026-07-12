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
    valid_papers: Optional[int] = 0  # 有效论文数
    category_taxonomy: Optional[Dict[str, Dict[str, List[str]]]] = None  # 分类体系

# 初始化搜索引擎
multi_search_agent = MultiSearchAgent()

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
            return await _advanced_search(request, filter_config=filter_config, sort_by=request.sort_by)
        else:
            # 使用简单搜索
            return await _simple_search(request)

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

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


def build_category_taxonomy(raw: Dict) -> Dict[str, Dict[str, List[str]]]:
    """
    按照检索时使用的查询/关键词对论文进行分类，按来源分组。

    支持两种输入结构：
    1. 已分组结构（来自 collect_tree_queries）：
       {"arxiv": {"query1": ["id1", ...]}, "openalex": {"keyword1": ["id2", ...]}}
    2. 平面结构（来自简单搜索的 query_results）：
       {"query1": [paper_dict, ...], "keyword1|keyword2": [paper_dict, ...]}

    会自动将 `|` 连接的复合关键词拆分为独立分类项。

    Returns:
        {
            "arxiv": {"query1": ["id1"], ...},
            "openalex": {"keyword1": ["id2"], ...},
            ...
        }
    """
    taxonomy: Dict[str, Dict[str, List[str]]] = {}

    # 判断输入结构：已分组还是平面
    first_val = next(iter(raw.values()), None)
    if isinstance(first_val, dict):
        # 已按来源分组的结构（来自 collect_tree_queries）
        grouped_input = raw
    else:
        # 平面结构（来自 query_results），放入 "queries" 组
        grouped_input = {"queries": raw}

    for group, queries in grouped_input.items():
        if group not in taxonomy:
            taxonomy[group] = {}

        expanded: Dict[str, List[str]] = {}
        for q, ids_or_papers in queries.items():
            # 提取论文ID（支持 [paper_dict, ...] 和 [id_str, ...] 两种格式）
            ids: List[str] = []
            for item in (ids_or_papers if isinstance(ids_or_papers, (list, tuple)) else []):
                if isinstance(item, dict):
                    pid = item.get('paper_id')
                    if pid:
                        ids.append(pid)
                elif isinstance(item, str):
                    ids.append(item)
            if not ids:
                continue
            # 去重
            ids = list(dict.fromkeys(ids))

            # 拆分 | 连接的复合关键词
            if '|' in q:
                parts = [p.strip() for p in q.split('|') if p.strip()]
                for part in parts:
                    if part in expanded:
                        expanded[part].extend(ids)
                    else:
                        expanded[part] = ids.copy()
            else:
                if q in expanded:
                    expanded[q].extend(ids)
                else:
                    expanded[q] = ids

        # 最终去重 + 按论文数降序排列
        for q, ids in expanded.items():
            expanded[q] = list(dict.fromkeys(ids))
        taxonomy[group] = dict(sorted(expanded.items(), key=lambda x: len(x[1]), reverse=True))

    return taxonomy


def collect_tree_queries(root, valid_ids: set) -> Dict[str, Dict[str, List[str]]]:
    """
    递归遍历搜索树，从每个节点中提取来源→查询→论文ID的映射。

    根据节点的 source 字段区分查询来源：
    - "arxiv" → arXiv 查询
    - "openalex" → OpenAlex 关键词
    - 其他 → 其他

    Args:
        root: SearchNode 根节点
        valid_ids: 有效的 paper_id 集合

    Returns:
        {"arxiv": {"query1": ["id1", "id2"], ...}, "openalex": {"keyword1": [...]}, ...}
    """
    result: Dict[str, Dict[str, List[str]]] = {}

    def _get_source_group(node_source) -> str:
        """根据节点 source 字段判断归属分组"""
        if isinstance(node_source, str):
            sl = node_source.lower()
            if 'arxiv' in sl:
                return 'arxiv'
            elif 'openalex' in sl:
                return 'openalex'
            elif 'pubmed' in sl:
                return 'pubmed'
        elif isinstance(node_source, (list, tuple)):
            for s in node_source:
                g = _get_source_group(s)
                if g != 'other':
                    return g
        return 'other'

    def _traverse(node):
        if node.query_str and node.docs:
            matched = []
            for doc in node.docs:
                if not isinstance(doc, dict):
                    continue
                pid = doc.get('paper_id') or doc.get('arxivId')
                if pid and pid in valid_ids:
                    matched.append(pid)
            if matched:
                group = _get_source_group(node.source)
                if group not in result:
                    result[group] = {}
                if node.query_str in result[group]:
                    result[group][node.query_str].extend(matched)
                else:
                    result[group][node.query_str] = matched
        for child in node.children:
            _traverse(child)

    _traverse(root)

    # 去重（每个组内的每个查询的论文ID去重）
    for group, queries in result.items():
        for q, ids in queries.items():
            result[group][q] = list(dict.fromkeys(ids))
    return result


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
        expanded_query_papers = {}  # 从搜索树中收集扩展查询→论文映射

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

                # 从搜索树中提取扩展查询及对应论文
                try:
                    valid_ids = set(papers_dict.keys())
                    if valid_ids and hasattr(search_agent, 'root') and search_agent.root:
                        tree_queries = collect_tree_queries(search_agent.root, valid_ids)
                        for group, queries in tree_queries.items():
                            if group not in expanded_query_papers:
                                expanded_query_papers[group] = {}
                            for q, ids in queries.items():
                                if q in expanded_query_papers[group]:
                                    expanded_query_papers[group][q].extend(ids)
                                else:
                                    expanded_query_papers[group][q] = ids
                except Exception as collect_error:
                    logger.error(f"Error collecting tree queries: {str(collect_error)}")

            except Exception as query_error:
                logger.error(f"Error processing query '{query}': {str(query_error)}")
                logger.error(f"Query error traceback: {traceback.format_exc()}")
                # 为失败的查询添加空结果
                all_results[query] = []
                query_source_map[query] = "error"

        # 构造响应
        category_taxonomy = build_category_taxonomy(expanded_query_papers if expanded_query_papers else all_results)
        response = SearchResponse(
            status="success",
            total_papers=len(all_papers),
            query_results=all_results,
            all_papers=all_papers,
            query_source_map=query_source_map,
            search_tree=search_trees,
            valid_papers=sum(1 for p in all_papers.values() if p.get('is_valid', False)),  # wsl-73
            category_taxonomy=category_taxonomy
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
        category_taxonomy = build_category_taxonomy(standardized_query_results)
        response = SearchResponse(
            status="success",
            total_papers=len(standardized_all_papers),
            query_results=standardized_query_results,
            all_papers=standardized_all_papers,
            query_source_map=query_source_map,
            search_tree=None,  # 简单搜索不生成搜索树
            valid_papers=sum(1 for p in all_papers.values() if p.get('is_valid', False)),  # wsl-73
            category_taxonomy=category_taxonomy
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