# coding:utf-8
# @FileName: citation_generator.py.py
# @Author  : BLC
# @Time: 2026/6/25 22:06
# @Project: SPAR-master
# @Function: 用于生成检索到论文的引用格式
# citation_generator.py
"""
论文引用格式生成器
支持 APA、MLA、Chicago、BibTeX、GB/T 7714 格式
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import re


class CitationGenerator:
    """统一引用格式生成器"""

    def __init__(self):
        self.default_style = "apa"

    def generate_all_citations(self, paper: Dict[str, Any]) -> Dict[str, str]:
        """
        为单篇论文生成所有格式的引用

        Args:
            paper: 论文信息字典

        Returns:
            包含所有引用格式的字典
        """
        return {
            "apa": self.format_apa(paper),
            "mla": self.format_mla(paper),
            "chicago": self.format_chicago(paper),
            "bibtex": self.generate_bibtex(paper),
            "gb7714": self.format_gb7714(paper)
        }

    def format_apa(self, paper: Dict[str, Any]) -> str:
        """
        生成 APA 7th Edition 格式

        格式: Author, A. A. (Year). Title of paper. Journal Name, Volume (Issue), Pages.
        """
        try:
            authors = self._format_authors_apa(self._get_authors(paper))
            year = self._get_year(paper)
            title = self._clean_title(paper.get("title", "Untitled"))
            journal = self._get_journal(paper)
            volume = self._get_volume(paper)
            issue = self._get_issue(paper)
            pages = self._get_pages(paper)
            doi = paper.get("doi", "")

            citation = f"{authors} ({year}). {title}."

            if journal:
                citation += f" {journal}"
                if volume:
                    citation += f", {volume}"
                if issue:
                    citation += f"({issue})"
                if pages:
                    citation += f", {pages}"
                citation += "."

            if doi:
                citation += f" https://doi.org/{doi}"

            return citation
        except Exception as e:
            print(f"APA format error: {e}")
            return paper.get("title", "Untitled")

    def format_mla(self, paper: Dict[str, Any]) -> str:
        """
        生成 MLA 9th Edition 格式

        格式: Author, A. A. "Title of Paper." Journal Name, vol. Volume, no. Issue, Year, pp. Pages.
        """
        try:
            authors = self._format_authors_mla(self._get_authors(paper))
            title = self._clean_title(paper.get("title", "Untitled"))
            journal = self._get_journal(paper)
            volume = self._get_volume(paper)
            issue = self._get_issue(paper)
            year = self._get_year(paper)
            pages = self._get_pages(paper)

            citation = f'{authors}. "{title}."'

            if journal:
                citation += f" {journal}"
                if volume:
                    citation += f", vol. {volume}"
                if issue:
                    citation += f", no. {issue}"
                if year:
                    citation += f", {year}"
                if pages:
                    citation += f", pp. {pages}"
                citation += "."
            else:
                if year:
                    citation += f" {year}."

            return citation
        except Exception as e:
            print(f"MLA format error: {e}")
            return paper.get("title", "Untitled")

    def format_chicago(self, paper: Dict[str, Any]) -> str:
        """
        生成 Chicago 17th Edition (Author-Date) 格式

        格式: Author, A. A. Year. "Title of Paper." Journal Name Volume (Issue): Pages.
        """
        try:
            authors = self._format_authors_chicago(self._get_authors(paper))
            year = self._get_year(paper)
            title = self._clean_title(paper.get("title", "Untitled"))
            journal = self._get_journal(paper)
            volume = self._get_volume(paper)
            issue = self._get_issue(paper)
            pages = self._get_pages(paper)

            citation = f'{authors}. {year}. "{title}."'

            if journal:
                citation += f" {journal}"
                if volume:
                    citation += f" {volume}"
                if issue:
                    citation += f", no. {issue}"
                if pages:
                    citation += f": {pages}"
                citation += "."
            else:
                citation += "."

            return citation
        except Exception as e:
            print(f"Chicago format error: {e}")
            return paper.get("title", "Untitled")

    def generate_bibtex(self, paper: Dict[str, Any]) -> str:
        """生成 BibTeX 格式"""
        try:
            # 生成 BibTeX key
            authors = self._get_authors(paper)
            first_author = authors[0].get("name", "Unknown") if authors else "Unknown"
            last_name = first_author.split(",")[0].strip().split()[-1] if "," in first_author else first_author.split()[
                -1] if first_author else "Unknown"
            year = self._get_year(paper)
            bibtex_key = f"{last_name}{year}".lower().replace(" ", "")

            # 清理特殊字符
            title = self._clean_title(paper.get("title", "Untitled"))
            journal = self._get_journal(paper)
            volume = self._get_volume(paper)
            issue = self._get_issue(paper)
            pages = self._get_pages(paper)
            doi = paper.get("doi", "")
            arxiv_id = paper.get("arxivId", "")

            # 如果包含 arxiv_id，使用 @article 或 @misc
            entry_type = "article" if journal else "misc"

            bibtex = f"""@{entry_type}{{{bibtex_key},
  author = {{{self._format_authors_bibtex(authors)}}},
  title = {{{title}}},"""

            if journal:
                bibtex += f"""
  journal = {{{journal}}},"""

            if volume:
                bibtex += f"""
  volume = {{{volume}}},"""

            if issue:
                bibtex += f"""
  number = {{{issue}}},"""

            if pages:
                bibtex += f"""
  pages = {{{pages}}},"""

            if year:
                bibtex += f"""
  year = {{{year}}},"""

            if doi:
                bibtex += f"""
  doi = {{{doi}}},"""

            if arxiv_id:
                bibtex += f"""
  eprint = {{{arxiv_id}}},
  archivePrefix = {{arXiv}},"""

            bibtex += """
}"""

            return bibtex
        except Exception as e:
            print(f"BibTeX format error: {e}")
            return f"@misc{{unknown,\n  title = {{{paper.get('title', 'Untitled')}}},\n}}"

    def format_gb7714(self, paper: Dict[str, Any]) -> str:
        """
        生成 GB/T 7714-2015 格式（中国国家标准）

        格式: 作者. 题名[J]. 期刊名, 出版年, 卷号(期号): 页码.
        """
        try:
            authors = self._format_authors_gb7714(self._get_authors(paper))
            title = self._clean_title(paper.get("title", "Untitled"))
            journal = self._get_journal(paper)
            year = self._get_year(paper)
            volume = self._get_volume(paper)
            issue = self._get_issue(paper)
            pages = self._get_pages(paper)

            citation = f"{authors}. {title}[J]."

            if journal:
                citation += f" {journal}"
                if year:
                    citation += f", {year}"
                if volume:
                    citation += f", {volume}"
                if issue:
                    citation += f"({issue})"
                if pages:
                    citation += f": {pages}"
                citation += "."
            else:
                if year:
                    citation += f" {year}."

            return citation
        except Exception as e:
            print(f"GB/T 7714 format error: {e}")
            return paper.get("title", "Untitled")

    # ==================== 辅助方法 ====================

    def _get_authors(self, paper: Dict) -> List[Dict]:
        """统一获取作者列表"""
        authors = paper.get("authors", [])
        if not authors:
            return []

        # 处理不同格式的作者
        if isinstance(authors[0], dict):
            return authors
        elif isinstance(authors[0], str):
            # 如果 author 是字符串，尝试解析
            return [{"name": a.strip()} for a in authors if a.strip()]
        return []

    def _get_year(self, paper: Dict) -> str:
        """获取年份"""
        year = paper.get("year", paper.get("publicationYear", ""))
        if year:
            # 尝试从日期字符串中提取年份
            match = re.search(r"(\d{4})", str(year))
            if match:
                return match.group(1)
        return "n.d."

    def _get_journal(self, paper: Dict) -> str:
        """获取期刊名称"""
        journal = paper.get("journal", {})
        if isinstance(journal, dict):
            return journal.get("name", "")
        return str(journal)

    def _get_volume(self, paper: Dict) -> str:
        """获取卷号"""
        journal = paper.get("journal", {})
        if isinstance(journal, dict):
            return str(journal.get("volume", ""))
        return ""

    def _get_issue(self, paper: Dict) -> str:
        """获取期号"""
        journal = paper.get("journal", {})
        if isinstance(journal, dict):
            return str(journal.get("issue", ""))
        return ""

    def _get_pages(self, paper: Dict) -> str:
        """获取页码"""
        pages = paper.get("pages", "")
        if pages:
            return str(pages)
        # 尝试从其他字段获取
        if "page" in paper:
            return str(paper.get("page", ""))
        return ""

    def _clean_title(self, title: str) -> str:
        """清理标题中的特殊字符"""
        if not title:
            return "Untitled"
        # 移除多余的空白
        title = re.sub(r"\s+", " ", title.strip())
        # 确保标题以大写字母开头（对于英文标题）
        if title and title[0].islower():
            title = title[0].upper() + title[1:]
        return title

    # ==================== 作者格式化方法 ====================

    def _format_authors_apa(self, authors: List[Dict]) -> str:
        """APA 格式作者：Smith, J. & Johnson, M. (最多6个，超过用et al.)"""
        if not authors:
            return "Unknown"
        names = self._extract_author_names(authors)
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} & {names[1]}"
        if len(names) <= 6:
            return ", ".join(names[:-1]) + f", & {names[-1]}"
        return f"{names[0]} et al."

    def _format_authors_mla(self, authors: List[Dict]) -> str:
        """MLA 格式作者：Smith, John, and Mary Johnson"""
        if not authors:
            return "Unknown"
        names = self._extract_author_names(authors)
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]}, and {names[1]}"
        return f"{names[0]}, et al."

    def _format_authors_chicago(self, authors: List[Dict]) -> str:
        """Chicago 格式作者：Smith, John, and Mary Johnson"""
        if not authors:
            return "Unknown"
        names = self._extract_author_names(authors)
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return f"{names[0]} et al."

    def _format_authors_bibtex(self, authors: List[Dict]) -> str:
        """BibTeX 格式作者：Smith, John and Johnson, Mary"""
        if not authors:
            return "Unknown"
        names = self._extract_author_names_bibtex(authors)
        return " and ".join(names)

    def _format_authors_gb7714(self, authors: List[Dict]) -> str:
        """GB/T 7714 格式作者：作者1, 作者2, 等"""
        if not authors:
            return "佚名"
        names = self._extract_author_names(authors)
        if len(names) <= 3:
            return ", ".join(names)
        return f"{names[0]}, {names[1]}, {names[2]}, 等"

    def _extract_author_names(self, authors: List[Dict]) -> List[str]:
        """提取作者名称（标准格式）"""
        names = []
        for author in authors:
            if isinstance(author, dict):
                name = author.get("name", author.get("full_name", ""))
                if not name and "first" in author and "last" in author:
                    name = f"{author['last']}, {author['first'][0]}."
                elif not name and "first_name" in author and "last_name" in author:
                    name = f"{author['last_name']}, {author['first_name'][0]}."
            else:
                name = str(author)
            if name:
                names.append(name)
        return names

    def _extract_author_names_bibtex(self, authors: List[Dict]) -> List[str]:
        """提取作者名称（BibTeX格式）"""
        names = []
        for author in authors:
            if isinstance(author, dict):
                name = author.get("name", author.get("full_name", ""))
                if not name and "first" in author and "last" in author:
                    name = f"{author['last']}, {author['first']}"
                elif not name and "first_name" in author and "last_name" in author:
                    name = f"{author['last_name']}, {author['first_name']}"
            else:
                name = str(author)
            if name:
                names.append(name)
        return names


# ==================== 单例实例 ====================

_citation_generator = None


def get_citation_generator() -> CitationGenerator:
    """获取引用生成器单例"""
    global _citation_generator
    if _citation_generator is None:
        _citation_generator = CitationGenerator()
    return _citation_generator
