package com.demo.trash;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

@Component
public class TrashClassifier {

    public static final String[] CLASSES = {"cardboard", "glass", "metal", "paper", "plastic", "trash"};

    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};
    private static final int RESIZE = 256;
    private static final int SIZE = 224;

    private static final Map<String, String> CN = new LinkedHashMap<>();
    private static final Map<String, String> CATEGORY = new LinkedHashMap<>();
    private static final Map<String, String> TIP = new LinkedHashMap<>();

    static {
        CN.put("cardboard", "纸箱"); CN.put("glass", "玻璃"); CN.put("metal", "金属");
        CN.put("paper", "纸类"); CN.put("plastic", "塑料"); CN.put("trash", "其他垃圾");
        CATEGORY.put("cardboard", "可回收物"); CATEGORY.put("glass", "可回收物");
        CATEGORY.put("metal", "可回收物"); CATEGORY.put("paper", "可回收物");
        CATEGORY.put("plastic", "可回收物"); CATEGORY.put("trash", "其他垃圾");
        TIP.put("cardboard", "纸箱 · 压扁再投，保持干燥");
        TIP.put("glass", "玻璃 · 玻璃桶，倒空、轻放");
        TIP.put("metal", "金属 · 金属罐，倒空");
        TIP.put("paper", "纸类 · 保持干燥");
        TIP.put("plastic", "塑料 · 倒空、简单冲洗");
        TIP.put("trash", "其他垃圾 · 投入其他桶");
    }

    private final OrtEnvironment env;
    private final OrtSession session;

    public TrashClassifier(@Value("${trash.model.path}") String modelPath) throws Exception {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath, new OrtSession.SessionOptions());
    }

    public static class Result {
        public String label;
        public String labelCn;
        public String category;
        public String tip;
        public double confidence;
        public double[] probs;
    }

    public Result classify(BufferedImage src) throws Exception {
        float[] data = preprocess(src);
        float[][] logits;
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), new long[]{1, 3, SIZE, SIZE})) {
            try (OrtSession.Result r = session.run(Collections.singletonMap("input", tensor))) {
                logits = (float[][]) r.get(0).getValue();
            }
        }
        double[] p = softmax(logits[0]);
        int top1 = argmax(p);
        double best = p[top1];
        double second = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < p.length; i++) {
            if (i == top1) continue;
            if (p[i] > second) second = p[i];
        }

        Result res = new Result();
        res.probs = p;
        if (best < 0.70 || (best - second) < 0.15) {
            res.label = "uncertain";
            res.labelCn = "不确定";
            res.category = "不确定";
            res.tip = "模型没把握，请交给督导员，或从下方 6 类里手动选。";
        } else {
            res.label = CLASSES[top1];
            res.labelCn = CN.get(res.label);
            res.category = CATEGORY.get(res.label);
            res.tip = TIP.get(res.label);
        }
        res.confidence = best;
        return res;
    }

    /** 预处理：缩放到短边 256（保持比例）→ 中心裁 224×224 → 按 ImageNet 均值方差做 CHW 归一化。 */
    private float[] preprocess(BufferedImage src) {
        int w = src.getWidth(), h = src.getHeight();
        double scale = (double) RESIZE / Math.min(w, h);
        int nw = (int) Math.round(w * scale);
        int nh = (int) Math.round(h * scale);
        BufferedImage resized = new BufferedImage(nw, nh, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(src, 0, 0, nw, nh, null);
        g.dispose();
        int x0 = (nw - SIZE) / 2, y0 = (nh - SIZE) / 2;
        BufferedImage crop = resized.getSubimage(x0, y0, SIZE, SIZE);
        int[] px = new int[SIZE * SIZE];
        crop.getRGB(0, 0, SIZE, SIZE, px, 0, SIZE);
        float[] data = new float[3 * SIZE * SIZE];
        for (int i = 0; i < px.length; i++) {
            int c = px[i];
            int r = (c >> 16) & 0xFF;
            int gg = (c >> 8) & 0xFF;
            int b = c & 0xFF;
            data[i] = (r / 255.0f - MEAN[0]) / STD[0];
            data[SIZE * SIZE + i] = (gg / 255.0f - MEAN[1]) / STD[1];
            data[2 * SIZE * SIZE + i] = (b / 255.0f - MEAN[2]) / STD[2];
        }
        return data;
    }

    private double[] softmax(float[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) max = Math.max(max, v);
        double sum = 0;
        double[] p = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            p[i] = Math.exp(logits[i] - max);
            sum += p[i];
        }
        for (int i = 0; i < p.length; i++) p[i] /= sum;
        return p;
    }

    private int argmax(double[] p) {
        int best = 0;
        for (int i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
        return best;
    }
}
