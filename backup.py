"""
备份管理脚本
用于创建和管理项目版本备份

使用方法:
    python backup.py create v1.1    # 创建 v1.1 版本备份
    python backup.py list           # 列出所有备份
    python backup.py restore v1.0   # 恢复到 v1.0 版本
"""
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path


class BackupManager:
    """备份管理器"""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir).resolve()
        self.backup_prefix = "backup_"
        
        # 需要备份的文件/目录
        self.include_patterns = [
            "src",
            "*.py",
            "*.md",
        ]
        
        # 排除的目录
        self.exclude_dirs = [
            "checkpoints",
            "checkpoints_pv",
            "checkpoints_pro",
            "checkpoints_rulefree",
            "metrics",
            "test_metrics",
            "__pycache__",
            ".git",
        ]
    
    def create_backup(self, version: str) -> str:
        """
        创建版本备份
        
        Args:
            version: 版本号，如 "v1.0", "v1.1"
            
        Returns:
            备份目录路径
        """
        date_str = datetime.now().strftime("%Y%m%d")
        backup_name = f"{self.backup_prefix}{version}_{date_str}"
        backup_path = self.project_dir / backup_name
        
        # 如果已存在，添加序号
        if backup_path.exists():
            i = 1
            while (self.project_dir / f"{backup_name}_{i}").exists():
                i += 1
            backup_name = f"{backup_name}_{i}"
            backup_path = self.project_dir / backup_name
        
        # 创建备份目录
        backup_path.mkdir(exist_ok=True)
        
        # 复制 src 目录
        src_dir = self.project_dir / "src"
        if src_dir.exists():
            shutil.copytree(
                src_dir, 
                backup_path / "src",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
        
        # 复制 Python 文件
        for py_file in self.project_dir.glob("*.py"):
            if py_file.name != "backup.py":  # 不备份自己
                shutil.copy2(py_file, backup_path)
        
        # 复制 Markdown 文件
        for md_file in self.project_dir.glob("*.md"):
            shutil.copy2(md_file, backup_path)
        
        # 创建版本信息文件
        version_file = backup_path / "VERSION.txt"
        with open(version_file, "w", encoding="utf-8") as f:
            f.write(f"Version: {version}\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Backup Path: {backup_path}\n")
        
        print(f"✅ Backup created: {backup_name}")
        return str(backup_path)
    
    def list_backups(self) -> list:
        """列出所有备份"""
        backups = []
        
        for item in self.project_dir.iterdir():
            if item.is_dir() and item.name.startswith(self.backup_prefix):
                version_file = item / "VERSION.txt"
                info = {"name": item.name, "path": str(item)}
                
                if version_file.exists():
                    with open(version_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if ":" in line:
                                key, value = line.strip().split(":", 1)
                                info[key.strip().lower()] = value.strip()
                
                backups.append(info)
        
        return sorted(backups, key=lambda x: x["name"])
    
    def restore_backup(self, version: str) -> bool:
        """
        恢复指定版本的备份
        
        Args:
            version: 版本号，如 "v1.0"
            
        Returns:
            是否成功
        """
        # 查找匹配的备份
        backups = self.list_backups()
        matching = [b for b in backups if version in b["name"]]
        
        if not matching:
            print(f"❌ No backup found for version: {version}")
            return False
        
        # 使用最新的匹配备份
        backup_info = matching[-1]
        backup_path = Path(backup_info["path"])
        
        print(f"Restoring from: {backup_info['name']}")
        
        # 先创建当前版本的临时备份
        temp_backup = self.create_backup("temp_before_restore")
        print(f"Current state backed up to: {temp_backup}")
        
        # 恢复 src 目录
        src_backup = backup_path / "src"
        if src_backup.exists():
            current_src = self.project_dir / "src"
            if current_src.exists():
                shutil.rmtree(current_src)
            shutil.copytree(src_backup, current_src)
        
        # 恢复 Python 文件
        for py_file in backup_path.glob("*.py"):
            shutil.copy2(py_file, self.project_dir)
        
        # 恢复 Markdown 文件
        for md_file in backup_path.glob("*.md"):
            shutil.copy2(md_file, self.project_dir)
        
        print(f"✅ Restored to: {backup_info['name']}")
        return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    manager = BackupManager()
    command = sys.argv[1].lower()
    
    if command == "create":
        if len(sys.argv) < 3:
            print("Usage: python backup.py create <version>")
            print("Example: python backup.py create v1.1")
            return
        version = sys.argv[2]
        manager.create_backup(version)
    
    elif command == "list":
        backups = manager.list_backups()
        if not backups:
            print("No backups found.")
        else:
            print("\nAvailable backups:")
            print("-" * 50)
            for b in backups:
                print(f"  {b['name']}")
                if 'version' in b:
                    print(f"    Version: {b['version']}")
                if 'date' in b:
                    print(f"    Date: {b['date']}")
            print("-" * 50)
    
    elif command == "restore":
        if len(sys.argv) < 3:
            print("Usage: python backup.py restore <version>")
            print("Example: python backup.py restore v1.0")
            return
        version = sys.argv[2]
        manager.restore_backup(version)
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
