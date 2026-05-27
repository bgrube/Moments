{
	gROOT->SetStyle("Modern");

	gStyle->SetPadTopMargin(0.100);
	gStyle->SetPadBottomMargin(0.160);
	gStyle->SetPadLeftMargin(0.145);
	gStyle->SetPadRightMargin(0.115);
	gStyle->SetCanvasDefH(600.0);
	gStyle->SetCanvasDefW(600.0);

	// no borders
	gStyle->SetCanvasBorderMode(0);
	gStyle->SetCanvasBorderSize(0);
	gStyle->SetPadBorderMode   (0);
	gStyle->SetFrameBorderMode (0);
	gStyle->SetTitleBorderSize (0);
	gStyle->SetLegendBorderSize(1);
	// all fill colors set to white; only pads and canvases are transparent
	// note that ROOT's PNG export does not write an alpha channel
	// PNGs with correct transparency must be created from PDFs, for example, using
	// mutool convert -O resolution=<dpi, e.g. 600>,alpha -o <out.png> <in.pdf>
	// or
	// mutool draw -r <dpi, e.g. 600> -c rgba -o <out.png> <in.pdf>
	// the latter command is preferred, as it does not append the page number to the output file name
	gStyle->SetFillColor      (kWhite);
	gStyle->SetFrameFillColor (kWhite);
	gStyle->SetTitleColor     (kWhite, "");
	gStyle->SetTitleFillColor (kWhite);
	gStyle->SetStatColor      (kWhite);
	gStyle->SetLegendFillColor(kWhite);
	TColor* newColor = new TColor(TColor::GetFreeColorIndex(), 1, 1, 1, "transparent_background", 0);
	int fullyTransparentColor = newColor->GetNumber();
	gStyle->SetPadColor       (fullyTransparentColor);
	gStyle->SetCanvasColor    (fullyTransparentColor);
	// gStyle->SetLegendFillColor(fullyTransparentColor);

	gStyle->SetLineColor(kBlack);
	gStyle->SetMarkerStyle(20);
	gStyle->SetMarkerSize(0.5);

	gStyle->SetStripDecimals(false);
	gStyle->SetTitleOffset(1.5,"X");
	gStyle->SetTitleOffset(1.8,"Y");
	gStyle->SetLabelSize(0.03, "Z");

	gStyle->SetOptTitle(true);
	gStyle->SetOptStat (true);
	gStyle->SetOptFit  (true);
	gStyle->SetOptDate (false);

	// palette for 2D plots
	// gStyle->SetPalette(kViridis);
	gStyle->SetPalette(kBird);
	gStyle->SetNumberContours(100);

	gStyle->SetHistMinimumZero(true);
	TTreeFormula::SetMaxima(100000,10000,10000);
	TTree::SetMaxTreeSize(2e12);
	TH1::SetDefaultSumw2(true);
	TH2::SetDefaultSumw2(true);
	TH3::SetDefaultSumw2(true);
	TGaxis::SetMaxDigits(3);  // restrict the number of digits in axis tick labels

	// gStyle->SetPaperSize(kA4);   // set A4 paper size
	gStyle->SetPaperSize(TStyle::kUSLetter);  // set letter paper size
}
