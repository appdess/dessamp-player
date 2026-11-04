import XCTest
import Vision
import ImageIO

final class Player: XCTestCase {
    let app = XCUIApplication(bundleIdentifier: "com.dessdynamics.dessamp")
    func tap(_ x: CGFloat, _ y: CGFloat) { app.coordinate(withNormalizedOffset:.init(dx:x,dy:y)).tap() }
    func words(heroOnly: Bool = false) throws -> [String] {
        let request = VNRecognizeTextRequest(); request.usesLanguageCorrection = false
        if heroOnly { request.regionOfInterest = CGRect(x: 0.05, y: 0.71, width: 0.90, height: 0.13) }
        // Screen PNGs carry their orientation as EXIF. The app snapshot can be
        // clipped to its previous dimensions while XCTest rotates the simulator.
        let source = try XCTUnwrap(CGImageSourceCreateWithData(XCUIScreen.main.screenshot().pngRepresentation as CFData, nil))
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let raw = properties?[kCGImagePropertyOrientation] as? UInt32 ?? 1
        let orientation = CGImagePropertyOrientation(rawValue: raw) ?? .up
        try VNImageRequestHandler(cgImage: XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil)),
                                  orientation: orientation).perform([request])
        return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
    }
    func choose(_ label: String) {
        tap(0.29, 0.872)
        let cell = app.tables.cells.containing(.staticText, identifier: label).firstMatch
        XCTAssertTrue(cell.waitForExistence(timeout: 5), "Missing preset: " + label)
        cell.tap()
    }
    func record(_ label: String) {
        let a = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        a.name = label; a.lifetime = .keepAlways; add(a)
    }
    func testPresetBank() throws {
        XCUIDevice.shared.orientation = .portrait
        app.launch()
        tap(0.38, 0.14)
        let label = "QA Lead " + UUID().uuidString.prefix(5)
        tap(0.15,0.934)
        choose("Model Default")
        XCTAssertTrue(try words().contains("5.0"))
        let start=app.coordinate(withNormalizedOffset:.init(dx:0.30,dy:0.479))
        start.press(forDuration:0.3,thenDragTo:app.coordinate(withNormalizedOffset:.init(dx:0.30,dy:0.429)))
        let edited=try words()
        XCTAssertFalse(edited.contains("5.0"), "Gain did not change: " + edited.joined(separator:"|"))
        let gain = try XCTUnwrap(edited.first { $0.range(of: #"^6\.\d$"#, options: .regularExpression) != nil }, edited.joined(separator:"|"))
        tap(0.80,0.872)
        let alert=app.alerts["Preset name"]
        XCTAssertTrue(alert.waitForExistence(timeout:5))
        alert.textFields.firstMatch.tap()
        alert.textFields.firstMatch.typeText(String(label))
        alert.buttons["Save"].tap()
        choose("Model Default")
        XCTAssertTrue(try words().contains("5.0"))
        choose(String(label))
        XCTAssertTrue(try words().contains(gain))
        tap(0.38,0.934)
        tap(0.29,0.872)
        XCTAssertTrue(app.tables.firstMatch.waitForExistence(timeout:3))
        XCTAssertFalse(app.tables.staticTexts[String(label)].exists, "FIRE preset leaked into FORGE")
        app.tables.staticTexts["Model Default"].tap()
        tap(0.15,0.934)
        app.terminate(); app.launch()
        choose(String(label))
        XCTAssertTrue(try words().contains(gain))
        XCUIDevice.shared.orientation = .landscapeLeft
        try expectPageText(gain)
        XCUIDevice.shared.orientation = .portrait
        XCTAssertTrue(try words().contains(gain))
        choose("Model Default")
        record("portrait")
        XCUIDevice.shared.orientation = .landscapeLeft
        record("landscape")
        XCUIDevice.shared.orientation = .portrait
        print("Saved and recalled", label, "gain", gain)
    }

    // A quick one-finger swipe, starting at a controlled point on the editor.
    func swipe(left: Bool, y: CGFloat, velocity: XCUIGestureVelocity = .fast) {
        let start = app.coordinate(withNormalizedOffset: .init(dx: left ? 0.73 : 0.27, dy: y))
        let end = app.coordinate(withNormalizedOffset: .init(dx: left ? 0.27 : 0.73, dy: y))
        start.press(forDuration: 0.01, thenDragTo: end, withVelocity: velocity, thenHoldForDuration: 0)
    }
    func expectPageText(_ label: String) throws {
        let deadline = Date().addingTimeInterval(4)
        var page = ""
        repeat {
            page = try words().joined(separator: "|").lowercased()
            if page.contains(label.lowercased()) { return }
        } while Date() < deadline
        XCTFail("Missing " + label + " in " + page)
    }
    func testSwipeAndCabinetShelf() throws {
        XCUIDevice.shared.orientation = .portrait
        app.launch()
        tap(0.38, 0.14)
        tap(0.15, 0.934)
        choose("Model Default")
        try expectPageText("Lead Gain")
        let knob = app.coordinate(withNormalizedOffset: .init(dx: 0.30, dy: 0.479))
        knob.press(forDuration: 0.01, thenDragTo: app.coordinate(withNormalizedOffset: .init(dx: 0.70, dy: 0.479)),
                   withVelocity: .fast, thenHoldForDuration: 0)
        try expectPageText("Lead Gain") // A horizontal knob drag must not change page.
        swipe(left: false, y: 0.28)
        try expectPageText("Tone")
        swipe(left: false, y: 0.14) // PRE is the first page.
        try expectPageText("Tone")
        record("pedals")
        swipe(left: true, y: 0.14)
        try expectPageText("Lead Gain")
        swipe(left: true, y: 0.28)
        try expectPageText("CABINETS")
        for (name, x, y) in [("DZL", 0.26, 0.65), ("Gleichrichter", 0.75, 0.65),
                              ("Green", 0.26, 0.78), ("SHFLD Mix", 0.75, 0.78)] {
            tap(x, y)
            let hero = try words(heroOnly: true).joined(separator: "|").lowercased()
            XCTAssertTrue(hero.contains(name.lowercased()), name + " hero: " + hero)
        }
        tap(0.26, 0.78) // Leave Green selected in the evidence capture.
        record("cabinets")
        swipe(left: true, y: 0.29)
        try expectPageText("Threshold")
        swipe(left: true, y: 0.14) // RIG is the last page.
        try expectPageText("Threshold")
        swipe(left: false, y: 0.14)
        try expectPageText("CABINETS")
        XCTAssertTrue(try words(heroOnly: true).joined().lowercased().contains("green"))
        swipe(left: false, y: 0.29)
        try expectPageText("Lead Gain")
        XCUIDevice.shared.orientation = .landscapeLeft
        try expectPageText("Lead Gain")
        // A deliberate pan across the full grille also works in landscape.
        swipe(left: true, y: 0.40, velocity: .slow)
        try expectPageText("LOAD IR")
        swipe(left: false, y: 0.40)
        try expectPageText("Lead Gain")
        XCUIDevice.shared.orientation = .portrait
    }

    func testTunerOpensAndReturnsToCurrentPage() throws {
        XCUIDevice.shared.orientation = .portrait
        app.launch()
        tap(0.62, 0.14)
        try expectPageText("CABINETS")
        tap(0.68, 0.09)
        try expectPageText("TUNER")
        try expectPageText("440 Hz")
        swipe(left: true, y: 0.50)
        try expectPageText("TUNER")
        record("tuner")
        tap(0.87, 0.385)
        try expectPageText("CABINETS")
    }

    func testUnsavedRigChoices() throws {
        XCUIDevice.shared.orientation = .portrait
        app.launch()
        tap(0.38, 0.14)
        tap(0.15, 0.934)
        choose("Model Default")
        let start = app.coordinate(withNormalizedOffset: .init(dx: 0.30,dy: 0.479))
        let end = app.coordinate(withNormalizedOffset: .init(dx: 0.30,dy: 0.429))
        start.press(forDuration: 0.3, thenDragTo: end)
        tap(0.62, 0.14)
        try expectPageText("CABINETS")
        tap(0.26, 0.78)
        tap(0.38, 0.14)
        try expectPageText("Lead Gain")
        tap(0.38, 0.934)
        try expectPageText("Save this sound")
        tap(0.19, 0.55)
        try expectPageText("Lead Gain")

        tap(0.38, 0.934)
        tap(0.81, 0.55)
        let alert = app.alerts["Preset name"]
        XCTAssertTrue(alert.waitForExistence(timeout:5))
        alert.buttons["Cancel"].tap()
        try expectPageText("Save this sound")
        tap(0.81, 0.55)
        XCTAssertTrue(alert.waitForExistence(timeout:5))
        let name = "Rig QA " + UUID().uuidString.prefix(6)
        alert.textFields.firstMatch.tap()
        alert.textFields.firstMatch.typeText(String(name))
        alert.buttons["Save"].tap()
        tap(0.15, 0.934)
        choose(String(name))
        tap(0.62, 0.14)
        XCTAssertTrue(try words(heroOnly:true).joined().lowercased().contains("green"))

        tap(0.38, 0.14)
        let previous = Set(try words())
        start.press(forDuration: 0.3, thenDragTo: end)
        let edited = try words()
        let savedGain = try XCTUnwrap(edited.first { !previous.contains($0) && $0.range(of: #"^\d+\.\d$"#, options: .regularExpression) != nil },
                                      edited.joined(separator:"|"))
        tap(0.38, 0.934)
        try expectPageText("Save this sound")
        tap(0.81, 0.55) // Named preset saves directly and completes the switch.
        tap(0.15, 0.934)
        choose(String(name))
        try expectPageText(savedGain)

        // Cabinet bypass by itself is a change worth protecting.
        tap(0.62, 0.14)
        tap(0.39, 0.52)
        try expectPageText("CAB OFF")
        choose("Model Default")
        try expectPageText("Save this sound")
        tap(0.50, 0.55)
        try expectPageText("CAB ON")
        tap(0.38, 0.14)
        try expectPageText("5.0")
    }

    func testFeedbackEntry() throws {
        XCUIDevice.shared.orientation = .portrait
        app.launch()
        tap(0.82, 0.09)
        try expectPageText("SETTINGS")
        tap(0.62, 0.255)
        try expectPageText("DessAmp feedback")
        try expectPageText("REPORT ISSUE")
        try expectPageText("EMAIL")
        tap(0.19, 0.55)
        try expectPageText("SETTINGS")
        tap(0.62, 0.255)
        try expectPageText("REPORT ISSUE")
        tap(0.81, 0.55)
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        XCTAssertTrue(safari.wait(for: .runningForeground, timeout: 12))
        app.activate()
        try expectPageText("SETTINGS")
    }
}
