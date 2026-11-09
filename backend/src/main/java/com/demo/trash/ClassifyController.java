package com.demo.trash;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.util.LinkedHashMap;
import java.util.Map;

@RestController
@CrossOrigin(origins = "*")
public class ClassifyController {

    private final TrashClassifier classifier;

    public ClassifyController(TrashClassifier classifier) {
        this.classifier = classifier;
    }

    @PostMapping("/api/classify")
    public Map<String, Object> classify(@RequestParam("file") MultipartFile file) throws Exception {
        BufferedImage img = ImageIO.read(file.getInputStream());
        if (img == null) {
            throw new IllegalArgumentException("无法读取图片，请上传 JPG/PNG");
        }
        TrashClassifier.Result r = classifier.classify(img);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("label", r.label);
        out.put("labelCn", r.labelCn);
        out.put("confidence", Math.round(r.confidence * 1000) / 1000.0);
        out.put("category", r.category);
        out.put("tip", r.tip);
        out.put("probs", r.probs);
        return out;
    }

    @GetMapping("/api/health")
    public Map<String, Object> health() {
        return Map.of("ok", true, "classes", TrashClassifier.CLASSES);
    }
}
