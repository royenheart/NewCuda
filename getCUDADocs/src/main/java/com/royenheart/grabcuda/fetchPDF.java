package com.royenheart.grabcuda;

import org.apache.commons.io.FileUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 获取CUDA Toolkit Documentation文档
 *
 * @author RoyenHeart
 */
public class fetchPDF {

    static private final String url = "https://docs.nvidia.com/cuda/index.html";
    static private final String rootUrl = "https://docs.nvidia.com/cuda/";
    static private final Pattern splita = Pattern.compile("<li><a.*</a></li>");
    static private final Pattern splitnoli = Pattern.compile("<a.*</a>");
    static private final Pattern getPDF = Pattern.compile("(?<=href=\").*pdf(?=\")");

    public static void main(String[] args) {
        getPatternValue getV = new getPatternValue();
        try {
            String line;
            URL net = new URL(url);
            BufferedReader reader = new BufferedReader(new InputStreamReader(net.openStream(), StandardCharsets.UTF_8));
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                Matcher geta = splita.matcher(line);
                if (geta.find()) {
                    line = geta.group();
                }
                Matcher getanoli = splitnoli.matcher(line);
                if (getanoli.find()) {
                    line = getanoli.group();
                }
                String href = getV.get_a_Href(line);
                String hrefName = getV.get_a_InnerHtml(line);
                if (href != null && hrefName != null) {
                    System.out.println("Enter " + href + ":\n" + hrefName + "\n");
                    URL conPage = new URL(rootUrl + href);
                    BufferedReader page = new BufferedReader(new InputStreamReader(conPage.openStream(), StandardCharsets.UTF_8));
                    while ((line = page.readLine()) != null) {
                        line = line.trim();
                        Matcher getPDFLink = getPDF.matcher(line);
                        if (getPDFLink.find()) {
                            line = getPDFLink.group();
                            line = line.replace("../", rootUrl);
                            String finalLine = line;
                            downloadThread newT = new downloadThread(hrefName, finalLine);
                            newT.run();
                            break;
                        }
                    }
                }
            }
            reader.close();
        } catch (MalformedURLException e) {
            System.err.println("CUDA web format err");
        } catch (IOException e) {
            System.err.println("CUDA web open failed");
        }
    }
}

/**
 * 文件下载进程
 */
class downloadThread implements Runnable {

    private final String hrefName;
    private final String finalLine;

    public downloadThread(String hrefName, String finalLine) {
        this.hrefName = hrefName;
        this.finalLine = finalLine;
    }

    @Override
    public void run() {
        try {
            System.out.println("Im now install " + hrefName + "\n");
            URL net1 = new URL(finalLine);
            File pdfFile = new File("./" + hrefName + ".pdf");
            FileUtils.copyURLToFile(net1, pdfFile);
        } catch (MalformedURLException e) {
            System.err.println("pdf link not find!" + finalLine);
            e.printStackTrace();
        } catch (IOException e) {
            System.err.println("pdf download failed!" + finalLine);
            e.printStackTrace();
        }
    }
}
