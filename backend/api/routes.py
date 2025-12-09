"""API 路由定义"""
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
import uuid
import json
from pathlib import Path
from datetime import datetime
import zipfile
import shutil

from backend.models.database import get_db, Project, Image, Class, Annotation
from backend.services.websocket_manager import websocket_manager
from backend.services.mqtt_service import mqtt_service
from backend.utils.yolo_export import YOLOExporter
from backend.config import settings
from PIL import Image as PILImage
import io


router = APIRouter()


# ========== Pydantic 模型 ==========

class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    class Config:
        from_attributes = True

    @classmethod
    def from_orm(cls, obj: Project):
        """从 ORM 对象创建响应模型"""
        return cls(
            id=obj.id,
            name=obj.name,
            description=obj.description or "",
            created_at=obj.created_at.isoformat() if obj.created_at else None,
            updated_at=obj.updated_at.isoformat() if obj.updated_at else None
        )


class ClassCreate(BaseModel):
    name: str
    color: str  # HEX 颜色
    shortcut_key: str = None


class AnnotationCreate(BaseModel):
    type: str  # bbox, polygon, keypoint
    data: dict  # 标注数据
    class_id: int


class AnnotationUpdate(BaseModel):
    data: dict = None
    class_id: int = None


# ========== 项目相关 ==========

@router.post("/projects", response_model=ProjectResponse)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """创建新项目"""
    project_id = str(uuid.uuid4())
    
    db_project = Project(
        id=project_id,
        name=project.name.strip(),
        description=project.description.strip() if project.description else ""
    )
    
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    
    # 创建项目目录
    (settings.DATASETS_ROOT / project_id / "raw").mkdir(parents=True, exist_ok=True)
    
    return ProjectResponse.from_orm(db_project)


@router.get("/projects", response_model=List[ProjectResponse])
def list_projects(db: Session = Depends(get_db)):
    """列出所有项目"""
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    return [ProjectResponse.from_orm(p) for p in projects]


@router.get("/projects/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)):
    """获取项目详情"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectResponse.from_orm(project)


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, db: Session = Depends(get_db)):
    """删除项目"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.delete(project)
    db.commit()
    
    # 删除项目目录
    project_dir = settings.DATASETS_ROOT / project_id
    if project_dir.exists():
        import shutil
        shutil.rmtree(project_dir)
    
    return {"message": "Project deleted"}


# ========== 类别相关 ==========

@router.post("/projects/{project_id}/classes")
def create_class(project_id: str, class_data: ClassCreate, db: Session = Depends(get_db)):
    """创建类别"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db_class = Class(
        project_id=project_id,
        name=class_data.name,
        color=class_data.color,
        shortcut_key=class_data.shortcut_key
    )
    
    db.add(db_class)
    db.commit()
    db.refresh(db_class)
    
    return db_class


@router.get("/projects/{project_id}/classes")
def list_classes(project_id: str, db: Session = Depends(get_db)):
    """列出项目所有类别"""
    classes = db.query(Class).filter(Class.project_id == project_id).all()
    return classes


@router.delete("/projects/{project_id}/classes/{class_id}")
def delete_class(project_id: str, class_id: int, db: Session = Depends(get_db)):
    """删除类别"""
    db_class = db.query(Class).filter(
        Class.id == class_id,
        Class.project_id == project_id
    ).first()
    
    if not db_class:
        raise HTTPException(status_code=404, detail="Class not found")
    
    # 检查是否有标注使用此类别
    annotation_count = db.query(Annotation).filter(Annotation.class_id == class_id).count()
    if annotation_count > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete class: {annotation_count} annotation(s) are using this class"
        )
    
    db.delete(db_class)
    db.commit()
    
    return {"message": "Class deleted"}


# ========== 图像相关 ==========

@router.post("/projects/{project_id}/images/upload")
async def upload_image(
    project_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """上传图像文件到项目"""
    # 校验项目是否存在
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # 校验文件类型
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {file_ext}。支持的格式: {', '.join(allowed_extensions)}"
        )
    
    try:
        # 读取文件内容
        file_content = await file.read()
        
        # 校验文件大小
        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"文件太大: {size_mb:.2f}MB (最大: {settings.MAX_IMAGE_SIZE_MB}MB)"
            )
        
        # 验证是否为有效图像并获取尺寸
        try:
            img = PILImage.open(io.BytesIO(file_content))
            img_width, img_height = img.size
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无效的图像文件: {str(e)}")
        
        # 生成存储路径
        project_dir = settings.DATASETS_ROOT / project_id / "raw"
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理文件名冲突和中文文件名
        original_filename = file.filename or f"image_{uuid.uuid4().hex[:8]}{file_ext}"
        # 处理中文文件名：使用UUID避免编码问题，但保留原始扩展名
        filename_stem = f"img_{uuid.uuid4().hex[:8]}"
        filename = f"{filename_stem}{file_ext}"
        file_path = project_dir / filename
        
        # 如果文件名冲突，添加时间戳
        counter = 0
        while file_path.exists():
            counter += 1
            filename = f"{filename_stem}_{counter}{file_ext}"
            file_path = project_dir / filename
        
        # 保存文件（如果图像格式需要转换，则在保存时转换）
        if img.mode != 'RGB' and file_ext in ['.jpg', '.jpeg']:
            # JPG格式需要RGB模式
            img_rgb = img.convert('RGB')
            img_rgb.save(file_path, 'JPEG', quality=95)
        else:
            # 其他格式直接保存原始内容
            file_path.write_bytes(file_content)
        
        # 生成相对路径（仅包含 raw/filename，不包含 project_id）
        relative_path = f"raw/{filename}"
        
        # 存入数据库（存储原始文件名和相对路径）
        db_image = Image(
            project_id=project_id,
            filename=original_filename,  # 存储原始文件名
            path=relative_path,  # 存储相对路径 raw/filename
            width=img_width,
            height=img_height,
            status="UNLABELED",
            source="UPLOAD"
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)
        
        # 通过 WebSocket 通知前端
        websocket_manager.broadcast_project_update(project_id, {
            "type": "new_image",
            "image_id": db_image.id,
            "filename": filename,
            "path": relative_path,
            "width": img_width,
            "height": img_height
        })
        
        return {
            "id": db_image.id,
            "filename": filename,
            "path": relative_path,
            "width": img_width,
            "height": img_height,
            "status": db_image.status,
            "message": "图像上传成功"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Upload] Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.get("/projects/{project_id}/images")
def list_images(project_id: str, db: Session = Depends(get_db)):
    """列出项目所有图像"""
    images = db.query(Image).filter(Image.project_id == project_id).order_by(Image.created_at.desc()).all()
    
    result = []
    for img in images:
        result.append({
            "id": img.id,
            "filename": img.filename,
            "path": img.path,
            "width": img.width,
            "height": img.height,
            "status": img.status,
            "created_at": img.created_at.isoformat() if img.created_at else None
        })
    
    return result


@router.get("/projects/{project_id}/images/{image_id}")
def get_image(project_id: str, image_id: int, db: Session = Depends(get_db)):
    """获取图像详情（含标注）"""
    image = db.query(Image).filter(
        Image.id == image_id,
        Image.project_id == project_id
    ).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # 获取标注
    annotations = db.query(Annotation).filter(Annotation.image_id == image_id).all()
    
    ann_list = []
    for ann in annotations:
        class_obj = db.query(Class).filter(Class.id == ann.class_id).first()
        ann_list.append({
            "id": ann.id,
            "type": ann.type,
            "data": json.loads(ann.data) if isinstance(ann.data, str) else ann.data,
            "class_id": ann.class_id,
            "class_name": class_obj.name if class_obj else None,
            "class_color": class_obj.color if class_obj else None
        })
    
    return {
        "id": image.id,
        "filename": image.filename,
        "path": image.path,
        "width": image.width,
        "height": image.height,
        "status": image.status,
        "annotations": ann_list
    }


@router.delete("/projects/{project_id}/images/{image_id}")
def delete_image(project_id: str, image_id: int, db: Session = Depends(get_db)):
    """删除图像"""
    image = db.query(Image).filter(
        Image.id == image_id,
        Image.project_id == project_id
    ).first()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # 先删除关联的标注数据，避免残留孤立记录
    annotations = db.query(Annotation).filter(Annotation.image_id == image_id).all()
    for ann in annotations:
        db.delete(ann)

    # 删除图像文件
    image_path = settings.DATASETS_ROOT / project_id / image.path
    if image_path.exists():
        try:
            image_path.unlink()
            print(f"[Delete] Deleted image file: {image_path}")
        except Exception as e:
            print(f"[Delete] Error deleting file {image_path}: {e}")
            # 继续删除数据库记录，即使文件删除失败
    
    # 删除数据库记录（级联删除标注）
    db.delete(image)
    db.commit()
    
    # 通过 WebSocket 通知前端
    websocket_manager.broadcast_project_update(project_id, {
        "type": "image_deleted",
        "image_id": image_id
    })
    
    return {"message": "Image deleted"}


# ========== 标注相关 ==========

@router.post("/images/{image_id}/annotations")
def create_annotation(image_id: int, annotation: AnnotationCreate, db: Session = Depends(get_db)):
    """创建标注"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    project_id = image.project_id
    
    db_annotation = Annotation(
        image_id=image_id,
        class_id=annotation.class_id,
        type=annotation.type,
        data=json.dumps(annotation.data)
    )
    
    db.add(db_annotation)
    
    # 更新图像状态
    was_unlabeled = image.status == "UNLABELED"
    image.status = "LABELED"
    
    db.commit()
    db.refresh(db_annotation)
    
    # 如果状态从 UNLABELED 变为 LABELED，通知前端更新图像列表
    if was_unlabeled:
        websocket_manager.broadcast_project_update(project_id, {
            "type": "image_status_updated",
            "image_id": image_id,
            "status": "LABELED"
        })
    
    return db_annotation


@router.put("/annotations/{annotation_id}")
def update_annotation(annotation_id: int, annotation: AnnotationUpdate, db: Session = Depends(get_db)):
    """更新标注"""
    db_ann = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not db_ann:
        raise HTTPException(status_code=404, detail="Annotation not found")
    
    image = db.query(Image).filter(Image.id == db_ann.image_id).first()
    project_id = image.project_id if image else None
    
    if annotation.data is not None:
        db_ann.data = json.dumps(annotation.data)
    
    if annotation.class_id is not None:
        db_ann.class_id = annotation.class_id
    
    # 确保图像状态为 LABELED（如果之前不是）
    if image and image.status != "LABELED":
        image.status = "LABELED"
        if project_id:
            websocket_manager.broadcast_project_update(project_id, {
                "type": "image_status_updated",
                "image_id": db_ann.image_id,
                "status": "LABELED"
            })
    
    db.commit()
    db.refresh(db_ann)
    
    return db_ann


@router.delete("/annotations/{annotation_id}")
def delete_annotation(annotation_id: int, db: Session = Depends(get_db)):
    """删除标注"""
    db_ann = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not db_ann:
        raise HTTPException(status_code=404, detail="Annotation not found")
    
    image_id = db_ann.image_id
    image = db.query(Image).filter(Image.id == image_id).first()
    project_id = image.project_id if image else None
    
    db.delete(db_ann)
    
    # 检查是否还有标注，如果没有则更新状态
    remaining = db.query(Annotation).filter(Annotation.image_id == image_id).count()
    status_changed = False
    if remaining == 0:
        if image and image.status == "LABELED":
            image.status = "UNLABELED"
            status_changed = True
    
    db.commit()
    
    # 通过 WebSocket 通知前端
    if project_id:
        from backend.services.websocket_manager import websocket_manager
        websocket_manager.broadcast_project_update(project_id, {
            "type": "annotation_deleted",
            "annotation_id": annotation_id,
            "image_id": image_id
        })
        
        # 如果状态改变，也通知图像状态更新
        if status_changed:
            websocket_manager.broadcast_project_update(project_id, {
                "type": "image_status_updated",
                "image_id": image_id,
                "status": "UNLABELED"
            })
    
    return {"message": "Annotation deleted"}


# ========== WebSocket ==========
# 注意：WebSocket 路由不在 router 中注册，需要在 main.py 中单独注册
# 这样路径就不会有 /api 前缀


# ========== 图像文件服务 ==========

@router.get("/images/{project_id}/{image_path:path}")
def get_image_file(project_id: str, image_path: str):
    """获取图像文件"""
    import os
    from pathlib import Path
    
    print(f"[Image] Request received: project_id={project_id}, image_path={image_path}")
    
    # image_path 应该是 raw/filename 格式
    # 移除可能的 project_id 前缀（兼容旧数据）
    if image_path.startswith(f"{project_id}/"):
        image_path = image_path[len(project_id) + 1:]
    
    # 确保路径以 raw/ 开头
    if not image_path.startswith("raw/"):
        # 如果路径不包含 raw/，可能是旧格式，尝试添加
        image_path = f"raw/{image_path}"
    
    # 构建文件路径
    file_path = settings.DATASETS_ROOT / project_id / image_path
    
    # 规范化路径，处理可能的路径遍历攻击
    try:
        resolved_path = file_path.resolve()
        datasets_root = settings.DATASETS_ROOT.resolve()
        # 确保解析后的路径在数据集根目录下
        resolved_path.relative_to(datasets_root)
    except ValueError:
        print(f"[Image] Security check failed: {resolved_path} not under {datasets_root}")
        raise HTTPException(status_code=403, detail="Access denied: Invalid path")
    
    print(f"[Image] Resolved path: {resolved_path}")
    print(f"[Image] Path exists: {resolved_path.exists()}")
    print(f"[Image] DATASETS_ROOT: {datasets_root}")
    
    if not resolved_path.exists():
        # 尝试列出目录内容以便调试
        project_dir = settings.DATASETS_ROOT / project_id / "raw"
        if project_dir.exists():
            files = list(project_dir.glob("*"))
            print(f"[Image] Files in raw dir: {[f.name for f in files]}")
        else:
            print(f"[Image] Raw directory does not exist: {project_dir}")
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path} (resolved: {resolved_path})")
    
    # 确保是文件而不是目录
    if not resolved_path.is_file():
        raise HTTPException(status_code=404, detail="Path is not a file")
    
    return FileResponse(str(resolved_path))


# ========== YOLO 导出 ==========

@router.post("/projects/{project_id}/export/yolo")
def export_yolo(project_id: str, db: Session = Depends(get_db)):
    """导出项目为 YOLO 格式"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # 获取所有图像和标注
    images = db.query(Image).filter(Image.project_id == project_id).all()
    classes = db.query(Class).filter(Class.project_id == project_id).all()
    
    # 构建导出数据
    project_data = {
        "id": project_id,
        "name": project.name,
        "classes": [{"id": c.id, "name": c.name, "color": c.color} for c in classes],
        "images": []
    }
    
    for img in images:
        annotations = db.query(Annotation).filter(Annotation.image_id == img.id).all()
        
        ann_list = []
        for ann in annotations:
            class_obj = db.query(Class).filter(Class.id == ann.class_id).first()
            ann_list.append({
                "id": ann.id,
                "type": ann.type,
                "data": json.loads(ann.data) if isinstance(ann.data, str) else ann.data,
                "class_name": class_obj.name if class_obj else None
            })
        
        project_data["images"].append({
            "id": img.id,
            "filename": img.filename,
            "path": img.path,
            "width": img.width,
            "height": img.height,
            "annotations": ann_list
        })
    
    # 导出
    output_dir = settings.DATASETS_ROOT / project_id / "yolo_export"
    result = YOLOExporter.export_project(project_data, output_dir, settings.DATASETS_ROOT)
    
    return {
        "message": "Export completed",
        "output_dir": str(output_dir.relative_to(settings.DATASETS_ROOT)),
        "images_count": result['images_count'],
        "classes_count": result['classes_count']
    }


@router.get("/projects/{project_id}/export/yolo/download")
def download_yolo_export(project_id: str, db: Session = Depends(get_db)):
    """下载 YOLO 格式数据集 zip 包"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    output_dir = settings.DATASETS_ROOT / project_id / "yolo_export"
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="YOLO export not found. Please export first.")
    
    # 创建临时 zip 文件
    zip_path = settings.DATASETS_ROOT / project_id / f"{project.name}_yolo_dataset.zip"
    
    def generate_zip():
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(output_dir)
                    zipf.write(file_path, arcname)
        
        with open(zip_path, 'rb') as f:
            yield from f
        
        # 清理临时文件
        if zip_path.exists():
            zip_path.unlink()
    
    return StreamingResponse(
        generate_zip(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={project.name}_yolo_dataset.zip"
        }
    )


@router.get("/projects/{project_id}/export/zip")
def export_dataset_zip(project_id: str, db: Session = Depends(get_db)):
    """导出完整数据集 zip 包（包含所有图像和标注）"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    images = db.query(Image).filter(Image.project_id == project_id).all()
    classes = db.query(Class).filter(Class.project_id == project_id).all()
    
    if not images:
        raise HTTPException(status_code=400, detail="No images in project")
    
    # 创建临时 zip 文件
    zip_path = settings.DATASETS_ROOT / project_id / f"{project.name}_dataset.zip"
    
    def generate_zip():
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 添加类别信息
            classes_info = {
                "classes": [{"id": c.id, "name": c.name, "color": c.color} for c in classes]
            }
            zipf.writestr("classes.json", json.dumps(classes_info, ensure_ascii=False, indent=2))
            
            # 添加图像和标注
            for img in images:
                # 添加图像文件
                img_path = settings.DATASETS_ROOT / project_id / img.path
                if img_path.exists():
                    zipf.write(img_path, f"images/{img.filename}")
                
                # 获取标注
                annotations = db.query(Annotation).filter(Annotation.image_id == img.id).all()
                if annotations:
                    ann_list = []
                    for ann in annotations:
                        class_obj = db.query(Class).filter(Class.id == ann.class_id).first()
                        ann_data = json.loads(ann.data) if isinstance(ann.data, str) else ann.data
                        ann_list.append({
                            "id": ann.id,
                            "type": ann.type,
                            "data": ann_data,
                            "class_id": ann.class_id,
                            "class_name": class_obj.name if class_obj else None
                        })
                    
                    # 保存标注为 JSON
                    ann_filename = Path(img.filename).stem + ".json"
                    zipf.writestr(f"annotations/{ann_filename}", json.dumps(ann_list, ensure_ascii=False, indent=2))
        
        with open(zip_path, 'rb') as f:
            yield from f
        
        # 清理临时文件
        if zip_path.exists():
            zip_path.unlink()
    
    return StreamingResponse(
        generate_zip(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={project.name}_dataset.zip"
        }
    )


# ========== MQTT 服务管理 ==========

@router.get("/mqtt/status")
def get_mqtt_status():
    """获取 MQTT 服务状态"""
    from backend.config import get_local_ip
    
    if settings.MQTT_USE_BUILTIN_BROKER:
        broker = get_local_ip()
        port = settings.MQTT_BUILTIN_PORT
        broker_type = "builtin"
    else:
        broker = settings.MQTT_BROKER
        port = settings.MQTT_PORT
        broker_type = "external"
    
    return {
        "enabled": settings.MQTT_ENABLED,
        "use_builtin": settings.MQTT_USE_BUILTIN_BROKER if settings.MQTT_ENABLED else False,
        "broker_type": broker_type if settings.MQTT_ENABLED else None,
        "connected": mqtt_service.is_connected if settings.MQTT_ENABLED else False,
        "broker": broker if settings.MQTT_ENABLED else None,
        "port": port if settings.MQTT_ENABLED else None,
        "topic": settings.MQTT_UPLOAD_TOPIC if settings.MQTT_ENABLED else None
    }


@router.post("/mqtt/test")
def test_mqtt_connection():
    """测试 MQTT 连接"""
    if not settings.MQTT_ENABLED:
        return {
            "success": False,
            "message": "MQTT 服务已禁用",
            "error": "MQTT_ENABLED is False"
        }
    
    import socket
    import time
    
    try:
        from backend.config import get_local_ip
        
        # 确定要测试的 Broker 地址
        if settings.MQTT_USE_BUILTIN_BROKER:
            broker_host = get_local_ip()
            broker_port = settings.MQTT_BUILTIN_PORT
        else:
            broker_host = settings.MQTT_BROKER
            broker_port = settings.MQTT_PORT
        
        # 测试 TCP 连接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((broker_host, broker_port))
        sock.close()
        
        if result == 0:
            # TCP 连接成功，尝试 MQTT 连接
            import paho.mqtt.client as mqtt
            import uuid
            
            test_client = None
            connection_result = {"success": False, "message": ""}
            
            def on_connect_test(client, userdata, flags, rc):
                connection_result["success"] = (rc == 0)
                if rc == 0:
                    connection_result["message"] = "MQTT 连接成功"
                else:
                    connection_result["message"] = f"MQTT 连接失败，错误代码: {rc}"
                client.disconnect()
            
            def on_connect_fail_test(client, userdata):
                connection_result["success"] = False
                connection_result["message"] = "MQTT 连接超时"
            
            try:
                # 明确指定使用 MQTT 3.1.1 协议（aMQTT broker 不支持 MQTT 5.0）
                test_client = mqtt.Client(
                    client_id=f"test_client_{uuid.uuid4().hex[:8]}",
                    protocol=mqtt.MQTTv311
                )
                test_client.on_connect = on_connect_test
                test_client.on_connect_fail = on_connect_fail_test
                
                # 内置 Broker 不需要认证，外部 Broker 才需要
                if not settings.MQTT_USE_BUILTIN_BROKER:
                    if settings.MQTT_USERNAME and settings.MQTT_PASSWORD:
                        test_client.username_pw_set(settings.MQTT_USERNAME, settings.MQTT_PASSWORD)
                
                test_client.connect(broker_host, broker_port, keepalive=5)
                test_client.loop_start()
                
                # 等待连接结果（最多等待 3 秒）
                timeout = 3
                elapsed = 0
                while elapsed < timeout and connection_result["message"] == "":
                    time.sleep(0.1)
                    elapsed += 0.1
                
                test_client.loop_stop()
                test_client.disconnect()
                
                if connection_result["message"] == "":
                    connection_result["message"] = "连接超时"
                
                return {
                    "success": connection_result["success"],
                    "message": connection_result["message"],
                    "broker": f"{broker_host}:{broker_port}",
                    "broker_type": "builtin" if settings.MQTT_USE_BUILTIN_BROKER else "external",
                    "tcp_connected": True
                }
            except Exception as e:
                if test_client:
                    try:
                        test_client.loop_stop()
                        test_client.disconnect()
                    except:
                        pass
                return {
                    "success": False,
                    "message": f"MQTT 连接测试失败: {str(e)}",
                    "broker": f"{broker_host}:{broker_port}",
                    "broker_type": "builtin" if settings.MQTT_USE_BUILTIN_BROKER else "external",
                    "tcp_connected": True
                }
        else:
            error_msg = f"无法连接到 MQTT Broker ({broker_host}:{broker_port})"
            error_detail = ""
            
            # macOS/Linux 错误代码
            if result == 61 or result == 111:  # ECONNREFUSED (macOS: 61, Linux: 111)
                error_msg += " - 连接被拒绝"
                error_detail = "MQTT Broker 可能未运行。请启动 MQTT Broker 服务。"
            elif result == 60 or result == 110:  # ETIMEDOUT (macOS: 60, Linux: 110)
                error_msg += " - 连接超时"
                error_detail = "网络连接超时，请检查网络和防火墙设置。"
            elif result == 64 or result == 113:  # EHOSTUNREACH (macOS: 64, Linux: 113)
                error_msg += " - 无法到达主机"
                error_detail = "无法到达 MQTT Broker 地址，请检查 Broker 地址配置。"
            elif result == 51:  # ENETUNREACH (macOS)
                error_msg += " - 网络不可达"
                error_detail = "网络不可达，请检查网络连接和 Broker 地址。"
            else:
                error_msg += f" - 错误代码: {result}"
                error_detail = "请检查 MQTT Broker 配置和运行状态。"
            
            return {
                "success": False,
                "message": error_msg,
                "detail": error_detail,
                "broker": f"{broker_host}:{broker_port}",
                "broker_type": "builtin" if settings.MQTT_USE_BUILTIN_BROKER else "external",
                "tcp_connected": False,
                "error": "TCP connection failed",
                "error_code": result
            }
    except socket.gaierror as e:
        return {
            "success": False,
            "message": f"无法解析 MQTT Broker 地址: {str(e)}",
            "broker": f"{settings.MQTT_BROKER}:{settings.MQTT_PORT}",
            "tcp_connected": False,
            "error": "DNS resolution failed"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"连接测试失败: {str(e)}",
            "broker": f"{settings.MQTT_BROKER}:{settings.MQTT_PORT}",
            "tcp_connected": False,
            "error": str(e)
        }

